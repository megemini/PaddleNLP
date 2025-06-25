import warnings
import paddle
from paddle import _C_ops
from paddle.base import core, framework
from paddle.base.framework import Variable, in_dynamic_or_pir_mode, in_pir_mode
from paddle.base.libpaddle import DataType
from typing import Dict, Any, Optional, Union, List, Iterable

class Apollo(paddle.optimizer.Optimizer):
    """Apollo optimizer implementation for PaddlePaddle."""
    def __init__(
        self,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-6,
        parameters=None,
        weight_decay=0.0,
        correct_bias=True,
        scale_front=False,
        disable_nl=False,
        rank=None,
        scale=None,
        scale_type=None,
        proj=None,
        update_proj_gap=None,
        proj_type=None,
        grad_clip=None,
        name=None
    ):
        if learning_rate < 0.0:
            raise ValueError(f"Invalid learning rate: {learning_rate} - should be >= 0.0")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {beta1} - should be in [0.0, 1.0)")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {beta2} - should be in [0.0, 1.0)")
        if not 0.0 <= epsilon:
            raise ValueError(f"Invalid epsilon value: {epsilon} - should be >= 0.0")
        
        if rank is not None:
            if scale is None:
                raise ValueError("scale must be specified when using low-rank projection")
            if scale_type not in ["channel", "tensor"]:
                raise ValueError("scale_type must be either 'channel' or 'tensor'")
            if proj not in ["random", "svd"]:
                raise ValueError("proj must be either 'random' or 'svd'")

        # Store optimizer defaults
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.correct_bias = correct_bias
        self.scale_front = scale_front
        self.disable_nl = disable_nl
        self.rank = rank
        self.scale = scale
        self.scale_type = scale_type
        self.proj = proj
        self.proj_type = proj_type

        # Initialize state dict
        self._param_state = {}

        # Handle parameters initialization
        params = self._handle_parameters(parameters) if parameters is not None else None

        # Initialize superclass with processed parameters
        super().__init__(
            learning_rate=learning_rate,
            parameters=params,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            name=name
        )

    def _handle_parameters(self, parameters):
        """Process input parameters into standard format."""
        if isinstance(parameters, paddle.framework.ParamBase):
            # Single parameter
            return [{'params': [parameters]}]
        elif isinstance(parameters, (list, tuple)):
            # Check if it's a list of parameters
            if all(isinstance(p, paddle.framework.ParamBase) for p in parameters):
                return [{'params': list(parameters)}]
            # List of parameter groups or mixed parameters/groups
            param_groups = []
            for p in parameters:
                if isinstance(p, dict):
                    if 'params' in p:
                        params = p['params']
                        if isinstance(params, paddle.framework.ParamBase):
                            p = dict(p, params=[params])
                        else:
                            p = dict(p, params=list(params))
                        param_groups.append(p)
                    else:
                        raise ValueError("param group must contain 'params' key")
                elif isinstance(p, paddle.framework.ParamBase):
                    param_groups.append({'params': [p]})
                else:
                    raise TypeError("params must be a Parameter or dict of Parameters")
            return param_groups
        else:
            raise TypeError("parameters must be a Parameter, list of Parameters, or list of param groups")

    def _add_param_group(self, param_group):
        """Add a param group to the optimizer's param_groups."""
        if isinstance(param_group, paddle.framework.ParamBase):
            param_group = {'params': [param_group]}
        elif not isinstance(param_group, dict):
            raise TypeError("param group must be a Parameter or dict")

        params = param_group.get('params', None)
        if params is None:
            raise ValueError("param group must contain 'params' key")

        if isinstance(params, paddle.framework.ParamBase):
            params = [params]
        elif isinstance(params, (list, tuple)):
            params = list(params)
        else:
            raise TypeError("params must be a Parameter or list of Parameters")

        for param in params:
            if not isinstance(param, paddle.framework.ParamBase):
                raise TypeError("optimizer can only optimize paddle.Parameters")
            if not param.trainable:
                warnings.warn("parameter is not trainable")

        # Create new group with defaults
        new_group = {
            'params': params,
            'rank': param_group.get('rank', self.rank),
            'scale': param_group.get('scale', self.scale),
            'scale_type': param_group.get('scale_type', self.scale_type),
            'proj': param_group.get('proj', self.proj),
            'scale_front': param_group.get('scale_front', self.scale_front)
        }

        self._param_groups.append(new_group)

    def _create_state(self, param):
        """Create state for a parameter."""
        state = {
            "step": paddle.zeros([1], dtype="int64"),
            "exp_avg": paddle.zeros_like(param),
            "exp_avg_sq": paddle.zeros_like(param)
        }

        if not self.disable_nl:
            state["scaled_grad"] = paddle.zeros([1], dtype=param.dtype)

        # Find param's group
        group = None
        for g in self._param_groups:
            if param in g['params']:
                group = g
                break

        # Setup low-rank state if needed
        if group and group['rank'] is not None:
            norm_dim = 0 if param.shape[0] < param.shape[1] else 1
            state.update({
                "norm_dim": norm_dim,
                "rank": group['rank'],
                "scale": group['scale'],
                "scale_type": group['scale_type'],
                "proj": group['proj'],
                "scale_front": group['scale_front']
            })

            if group['proj'] == 'random':
                matrix_shape = [param.shape[norm_dim], group['rank']]
                std = 1.0 / paddle.sqrt(paddle.to_tensor(group['rank']))
                proj_matrix = paddle.normal(mean=0.0, std=std, shape=matrix_shape)
                state["proj_matrix"] = proj_matrix.astype(param.dtype)

        return state

    def _get_state(self, param):
        """Get or create state for a parameter."""
        param_id = id(param)
        if param_id not in self._param_state:
            self._param_state[param_id] = self._create_state(param)
        return self._param_state[param_id]

    @paddle.no_grad()
    def step(self):
        """Performs a single optimization step."""
        for param in self._parameter_list:
            if not param.trainable or param.grad is None:
                continue

            if param.grad.is_sparse:
                raise RuntimeError("Apollo doesn't support sparse gradients")

            grad = param.grad
            state = self._get_state(param)

            # Get state values
            step = state["step"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            # Update step
            step += 1

            # Project gradient if using low-rank optimization
            proj_grad = grad
            if "rank" in state:
                norm_dim = state["norm_dim"]
                proj_matrix = state["proj_matrix"]
                if norm_dim == 0:
                    proj_grad = paddle.matmul(grad, proj_matrix)
                    proj_grad = paddle.matmul(proj_grad, proj_matrix.t())
                else:
                    proj_grad = paddle.matmul(proj_matrix.t(), grad)
                    proj_grad = paddle.matmul(proj_matrix, proj_grad)

            # Update moment estimates
            exp_avg = self.beta1 * exp_avg + (1 - self.beta1) * proj_grad
            exp_avg_sq = self.beta2 * exp_avg_sq + (1 - self.beta2) * paddle.square(proj_grad)

            if self.correct_bias:
                # Bias corrections
                bias_correction1 = 1 - self.beta1 ** step.item()
                bias_correction2 = 1 - self.beta2 ** step.item()
            else:
                bias_correction1 = 1
                bias_correction2 = 1

            # Compute adaptive learning rate
            denom = paddle.sqrt(exp_avg_sq / bias_correction2) + self.epsilon

            # Calculate scaled gradient
            if "rank" in state:
                norm_grad = exp_avg / denom
                if state["scale_type"] == "channel":
                    grad_scaling_factor = (
                        paddle.norm(norm_grad, axis=norm_dim, keepdim=True) /
                        (paddle.norm(proj_grad, axis=norm_dim, keepdim=True) + self.epsilon)
                    )
                    if norm_dim == 1:
                        grad_scaling_factor = grad_scaling_factor.unsqueeze(1)
                else:  # tensor
                    grad_scaling_factor = (
                        paddle.norm(norm_grad) /
                        (paddle.norm(proj_grad) + self.epsilon)
                    )
                
                scaled_grad = proj_grad * grad_scaling_factor

                if state["scale_front"]:
                    scaled_grad *= paddle.sqrt(paddle.to_tensor(state["scale"]))

                if not self.disable_nl:
                    scaled_grad_norm = paddle.norm(scaled_grad)
                    if "scaled_grad" in state:
                        prev_norm = state["scaled_grad"]
                        ratio = scaled_grad_norm / (prev_norm + self.epsilon)
                        limiter = paddle.maximum(ratio, paddle.to_tensor(1.01)) / 1.01
                        scaled_grad = scaled_grad / limiter
                        state["scaled_grad"] = scaled_grad_norm / limiter
                    else:
                        state["scaled_grad"] = scaled_grad_norm

                norm_grad = scaled_grad

                if not state["scale_front"]:
                    norm_grad *= paddle.sqrt(paddle.to_tensor(state["scale"]))
            else:
                norm_grad = exp_avg / denom

            # Apply weight decay
            if self.weight_decay > 0:
                param.set_value(param * (1 - self.learning_rate * self.weight_decay))

            # Update parameter
            step_size = self.learning_rate / bias_correction1
            param.set_value(param - step_size * norm_grad)

            # Update state
            state["step"] = step
            state["exp_avg"] = exp_avg
            state["exp_avg_sq"] = exp_avg_sq

        self._learning_rate = self.learning_rate
