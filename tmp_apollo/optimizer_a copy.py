import warnings
import numpy as np
import paddle
from paddle import _C_ops, pir
from paddle.base import core, framework
from paddle.base.framework import Parameter, Variable, in_dynamic_or_pir_mode, in_pir_mode
from paddle.base.libpaddle import DataType
from paddle.optimizer.adamw import AdamW
from collections import defaultdict

class Apollo(AdamW):
    """
    The Apollo optimizer from paper `Apollo: An Adaptive Parameter-wise Diagonal Quasi-Newton Method for Nonconvex Stochastic Optimization`.

    Args:
        parameters (list|tuple): List/Tuple of parameter groups to optimize.
        learning_rate (float, optional): The learning rate. Default: 0.001.
        beta1 (float, optional): The exponential decay rate for the 1st moment estimates. Default: 0.9.
        beta2 (float, optional): The exponential decay rate for the 2nd moment estimates. Default: 0.999.
        epsilon (float, optional): A small constant for numerical stability. Default: 1e-8.
        weight_decay (float, optional): The weight decay coefficient. Default: 0.0.
        scale_front (bool, optional): Whether to apply scaling before projection. Default: False.
        disable_nl (bool, optional): Whether to disable norm limiter. Default: False.
        name (str, optional): The name of the optimizer. Default: None.

    Note:
        Additional parameters like rank, proj_type, scale etc. are set through parameter groups.
    """
    def __init__(self,
                 parameters=None,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 weight_decay=0.0,
                 scale_front=False,
                 disable_nl=False,
                 name=None,
                 **defaults):

        # Store Apollo-specific settings
        self.scale_front = scale_front
        self.disable_nl = disable_nl
        self._projectors = {}  # Store projectors for each parameter
        self._step_count = 0

        # Process parameter groups and extract all parameters for parent class
        processed_params = []
        self._apollo_param_groups = []  # Store our own parameter groups

        if parameters is not None:
            if isinstance(parameters, paddle.Tensor):
                raise TypeError(
                    "`parameters` argument should be an iterable of Tensors, got Tensor"
                )

            # Handle parameter groups
            if isinstance(parameters, list) and len(parameters) > 0 and isinstance(parameters[0], dict):
                for group_idx, group in enumerate(parameters):
                    # Extract parameters from this group
                    group_params = group['params']
                    for param in group_params:
                        processed_params.append(param)

                    # Store the group information for Apollo
                    apollo_group = {
                        'params': group_params,
                        'rank': group.get('rank', None),
                        'update_proj_gap': group.get('update_proj_gap', 1),
                        'scale': group.get('scale', 1.0),
                        'proj_type': group.get('proj_type', 'std'),
                        'proj': group.get('proj', 'random'),
                        'scale_type': group.get('scale_type', 'tensor'),
                    }
                    self._apollo_param_groups.append(apollo_group)
            else:
                # Single parameter list - create default group
                processed_params = list(parameters)
                default_group = {
                    'params': processed_params,
                    'rank': None,
                    'update_proj_gap': 1,
                    'scale': 1.0,
                    'proj_type': 'std',
                    'proj': 'random',
                    'scale_type': 'tensor',
                }
                self._apollo_param_groups.append(default_group)

        # Initialize parent class with all parameters
        super().__init__(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            parameters=processed_params,
            weight_decay=weight_decay,
            name=name)

        self._already_create_accumulator = set()



    def _initialize_projector(self, group, param):
        """Create a projector for the given parameter."""
        if "rank" not in group:
            return None

        if group["proj"] == 'random':
            return RandomProjector(
                rank=group["rank"],
                shape=None,
                update_proj_gap=group["update_proj_gap"],
                scale=group["scale"],
                proj_type=group["proj_type"],
                seed=0  # Use fixed seed, rely on global seed setting
            )
        elif group["proj"] == 'svd':
            return SVDProjector(
                rank=group["rank"],
                shape=None,
                update_proj_gap=group["update_proj_gap"],
                scale=group["scale"],
                proj_type=group["proj_type"]
            )
        else:
            raise ValueError("Invalid projector type specified in group")

    def _create_accumulators(self, block, parameters):
        """Create accumulators needed by Apollo optimizer."""
        assert isinstance(block, (framework.Block, pir.Block))
        if isinstance(parameters, dict):
            parameters = self._update_param_group(parameters)

        # Create accumulator tensors for first and second moments
        for p in parameters:
            if p.name in self._already_create_accumulator:
                continue

            # Find which Apollo parameter group this parameter belongs to
            group = None
            for apollo_group in self._apollo_param_groups:
                # Use parameter name for comparison instead of direct tensor comparison
                param_names = [param.name for param in apollo_group['params']]
                if p.name in param_names:
                    group = apollo_group
                    break

            if group is None:
                # Default group if not found
                group = {
                    'params': [p],
                    'rank': None,
                    'update_proj_gap': 1,
                    'scale': 1.0,
                    'proj_type': 'std',
                    'proj': 'random',
                    'scale_type': 'tensor',
                }

            # Initialize projector if needed and store it
            if group["rank"] is not None:
                projector = self._initialize_projector(group, p)
                self._projectors[p.name] = projector

            if self._multi_precision and self._is_dtype_fp16_or_bf16(p.dtype):
                master_p = self._create_master_weight(p)
                self._add_moments_pows(master_p)
                self._already_create_accumulator.add(p.name)
                continue

            if self._is_dtype_fp16_or_bf16(p.dtype) and not self._multi_precision:
                warnings.warn(
                    "Accumulating with FP16 or BF16 in optimizer can lead to poor accuracy or slow convergence."
                    "Consider using multi_precision=True option of the optimizer."
                )
            self._add_moments_pows(p)
            self._already_create_accumulator.add(p.name)
    
    def _add_moments_pows(self, p):
        """Add all necessary accumulators for parameter."""
        acc_dtype = p.dtype
        if self._is_dtype_fp16_or_bf16(acc_dtype) and not self._use_lowprecision_moment:
            acc_dtype = DataType.FLOAT32 if in_pir_mode() else core.VarDesc.VarType.FP32

        # For Apollo, we need to handle accumulator shapes differently
        # Find which Apollo parameter group this parameter belongs to
        group = None
        for apollo_group in self._apollo_param_groups:
            param_names = [param.name for param in apollo_group['params']]
            if p.name in param_names:
                group = apollo_group
                break

        # Add moment accumulators - always create standard accumulators
        # We'll handle shape adjustments dynamically in apollo_python
        self._add_accumulator(self._moment1_acc_str, p, dtype=acc_dtype)
        self._add_accumulator(self._moment2_acc_str, p, dtype=acc_dtype)

        # Add beta power accumulators
        self._add_accumulator(
            name=self._beta1_pow_acc_str,
            param=p,
            dtype=acc_dtype,
            fill_value=0.9 if isinstance(self._beta1, (Variable, paddle.Tensor)) else self._beta1,
            shape=[1],
            type=core.VarDesc.VarType.DENSE_TENSOR,
            device="cpu",
        )
        self._add_accumulator(
            name=self._beta2_pow_acc_str,
            param=p,
            dtype=acc_dtype,
            fill_value=0.999 if isinstance(self._beta2, (Variable, paddle.Tensor)) else self._beta2,
            shape=[1],
            type=core.VarDesc.VarType.DENSE_TENSOR,
            device="cpu",
        )

        # Add norm limiter accumulator if needed
        if not self.disable_nl:
            self._add_accumulator('scaled_grad_norm', p, dtype=acc_dtype, shape=[1])
        
    def _append_optimize_op(self, block, param_and_grad):
        """Implement optimization step for Apollo."""
        assert isinstance(block, (framework.Block, pir.Block))
        if isinstance(param_and_grad, dict):
            param_and_grad = self._update_param_group(param_and_grad)

        param, grad = param_and_grad

        if grad is None:
            return

        # Whether we should do weight decay for the parameter
        with_decay = True
        if self._apply_decay_param_fun is not None and not self._apply_decay_param_fun(param.name):
            with_decay = False

        # Get learning rate
        lr = self._create_param_lr(param_and_grad)

        # Find which Apollo parameter group this parameter belongs to
        group = None
        for apollo_group in self._apollo_param_groups:
            # Use parameter name for comparison instead of direct tensor comparison
            param_names = [p.name for p in apollo_group['params']]
            if param.name in param_names:
                group = apollo_group
                break

        if group is None:
            # Default group if not found
            group = {
                'params': [param],
                'rank': None,
                'update_proj_gap': 1,
                'scale': 1.0,
                'proj_type': 'std',
                'proj': 'random',
                'scale_type': 'tensor',
            }

        # Get moment accumulators - always use standard accumulators
        moment1 = self._get_accumulator_master(self._moment1_acc_str, param)
        moment2 = self._get_accumulator_master(self._moment2_acc_str, param)
        beta1_pow = self._get_accumulator_master(self._beta1_pow_acc_str, param)
        beta2_pow = self._get_accumulator_master(self._beta2_pow_acc_str, param)

        # Find master weight for multi-precision training
        find_master = self._multi_precision and self._is_dtype_fp16_or_bf16(param.dtype)
        master_weight = self._master_weights[param.name] if find_master else None

        # Create the apollo optimize op
        if in_dynamic_or_pir_mode():
            lr_ratio_ = 1.0 if self._lr_ratio is None else self._lr_ratio(param)
            _beta1 = self._beta1 if not isinstance(self._beta1, Variable) else self._beta1.item(0)
            _beta2 = self._beta2 if not isinstance(self._beta2, Variable) else self._beta2.item(0)
            found_inf = self._get_auxiliary_var("found_inf") if in_pir_mode() else None

            self.apollo_python(
                param,
                grad,
                lr,
                beta1_pow,
                beta2_pow,
                master_weight,
                found_inf,
                _beta1,
                _beta2,
                self._epsilon,
                lr_ratio_,
                self._weight_decay,
                with_decay,
                find_master,
                param.name,
                group,
            )
            return None
        else:
            raise NotImplementedError("Not implemented for static graph mode.")
            
    def apollo_python(
        self,
        param,
        grad,
        learning_rate,
        beta1_pow,
        beta2_pow,
        master_weight,
        skip_update,
        beta1,
        beta2,
        epsilon,
        lr_ratio,
        coeff,
        with_decay,
        multi_precision,
        name,
        group,
    ):
        """Implement Apollo optimization algorithm."""
        if skip_update:
            return

        # Apply weight decay if needed
        if not with_decay:
            coeff = 0.0
        if not multi_precision:
            master_weight = None

        # Get effective learning rate
        lr = learning_rate * lr_ratio

        # Use master weight if available
        if master_weight is not None:
            p = master_weight
        else:
            p = param

        # Apply weight decay
        if coeff > 0:
            p *= (1.0 - lr * coeff)

        # Increment step count
        self._step_count += 1

        # Store original gradient for scaling calculations
        original_grad = grad

        # Get moment accumulators
        moment1 = self._get_accumulator_master(self._moment1_acc_str, param)
        moment2 = self._get_accumulator_master(self._moment2_acc_str, param)

        # Project gradient if needed
        if param.name in self._projectors and len(grad.shape) > 1:
            projector = self._projectors[param.name]
            grad = projector.project(grad, self._step_count)

        # Keep all accumulators in original parameter shape
        # We'll handle projection in the computation logic

        # Handle moment updates based on whether parameter has projection
        if param.name in self._projectors and "rank" in group and len(original_grad.shape) > 1:
            # For parameters with projection, we need special handling
            projector = self._projectors[param.name]

            # Project moment1 to low-rank space for update
            if moment1.shape == original_grad.shape:
                # moment1 is in original space, project it
                projected_moment1 = projector.project(moment1, self._step_count)
            else:
                # moment1 is already in projected space
                projected_moment1 = moment1

            # Update first moment in projected space
            projected_mom1 = projected_moment1 * beta1 + (1.0 - beta1) * grad

            # Project back to original space
            mom1 = projector.project_back(projected_mom1)

            # Update second moment based on scale_type
            # For parameters with projection, we need to handle moment2 carefully
            if group['scale_type'] == 'channel':
                norm_dim = 0 if original_grad.shape[0] < original_grad.shape[1] else 1
                if norm_dim == 0:
                    mom2_update = (grad * grad).mean(axis=0, keepdim=True)
                    # Expand to match original gradient shape for storage
                    mom2_update_full = mom2_update.expand(original_grad.shape)
                else:
                    mom2_update = (grad * grad).mean(axis=1, keepdim=True)
                    # Expand to match original gradient shape for storage
                    mom2_update_full = mom2_update.expand(original_grad.shape)
                mom2 = moment2 * beta2 + (1.0 - beta2) * mom2_update_full
            else:  # tensor
                mom2_update = (grad * grad).mean()
                # Expand to match original gradient shape for storage
                mom2_update_full = paddle.full_like(original_grad, mom2_update)
                mom2 = moment2 * beta2 + (1.0 - beta2) * mom2_update_full
        else:
            # Standard case - no projection
            mom1 = moment1 * beta1 + (1.0 - beta1) * grad
            mom2 = moment2 * beta2 + (1.0 - beta2) * grad * grad

        # Compute bias correction (similar to torch version)
        bias_correction1 = 1.0 - beta1_pow
        bias_correction2 = 1.0 - beta2_pow

        # Compute adaptive learning rate and normalized gradient
        if param.name in self._projectors and "rank" in group and len(original_grad.shape) > 1:
            # For parameters with projection, compute in projected space
            projector = self._projectors[param.name]
            projected_mom1 = projector.project(mom1, self._step_count)

            # Compute denom based on scale_type
            if group['scale_type'] == 'channel':
                norm_dim = 0 if original_grad.shape[0] < original_grad.shape[1] else 1
                if norm_dim == 0:
                    # Extract the channel-wise moment2 values
                    mom2_channel = mom2[0:1, :]  # Take first row as representative
                    denom = mom2_channel.sqrt() / bias_correction2.sqrt() + epsilon
                else:
                    # Extract the channel-wise moment2 values
                    mom2_channel = mom2[:, 0:1]  # Take first column as representative
                    denom = mom2_channel.sqrt() / bias_correction2.sqrt() + epsilon
            else:  # tensor
                # Use scalar moment2 value
                mom2_scalar = mom2.mean()
                denom = mom2_scalar.sqrt() / bias_correction2.sqrt() + epsilon

            # Compute normalized gradient in projected space
            norm_grad = projected_mom1 / denom
        else:
            # Standard case
            denom = mom2.sqrt() / bias_correction2.sqrt() + epsilon
            norm_grad = mom1 / denom

        # Apply Apollo scaling if using low-rank projection
        if param.name in self._projectors and "rank" in group and len(original_grad.shape) > 1:
            if group['scale_type'] == 'channel':
                norm_dim = 0 if original_grad.shape[0] < original_grad.shape[1] else 1
                grad_scaling_factor = (
                    paddle.norm(norm_grad, axis=norm_dim, keepdim=True) /
                    (paddle.norm(grad, axis=norm_dim, keepdim=True) + epsilon)
                )
                if norm_dim == 1:
                    grad_scaling_factor = grad_scaling_factor.unsqueeze(1)
            else:  # tensor
                grad_scaling_factor = (
                    paddle.norm(norm_grad) /
                    (paddle.norm(grad) + epsilon)
                )


            # Scale original gradient
            scaled_grad = original_grad * grad_scaling_factor

            if self.scale_front:
                scaled_grad *= np.sqrt(group['scale'])

            # Apply norm limiter if enabled
            if not self.disable_nl:
                try:
                    scaled_grad_norm_acc = self._get_accumulator('scaled_grad_norm', param)
                    scaled_grad_norm = paddle.norm(scaled_grad)
                    prev_norm = scaled_grad_norm_acc[0]

                    limiter = paddle.maximum(
                        scaled_grad_norm / (prev_norm + epsilon),
                        paddle.to_tensor(1.01)
                    ) / 1.01
                    scaled_grad = scaled_grad / limiter
                    scaled_grad_norm = scaled_grad_norm / limiter
                    scaled_grad_norm_acc[0] = scaled_grad_norm
                except:
                    pass  # Skip norm limiter if accumulator not found

            if not self.scale_front:
                scaled_grad *= np.sqrt(group['scale'])

            norm_grad = scaled_grad

        # Update parameter
        p += (norm_grad * (-(lr / bias_correction1)))

        # Update master weight and param if needed
        if master_weight is not None:
            master_weight[:] = p
            param[:] = p.astype(param.dtype)
        else:
            param[:] = p

        # Update accumulators in-place
        moment1[:] = mom1
        moment2[:] = mom2
        beta1_pow[:] = beta1 * beta1_pow[:]
        beta2_pow[:] = beta2 * beta2_pow[:]

def stable_randn(
    shape,
    seed=None,
    device=None,
    dtype=None
):
    """
    Generates a stable random tensor.

    Args:
        shape: Shape of the tensor.
        seed: Random seed for reproducibility (ignored, use global seed).
        device: Device to generate the tensor on (ignored in Paddle).
        dtype: Data type of the tensor.

    Returns:
        paddle.Tensor: Generated random tensor.
    """
    # Generate random tensor using global seed
    return paddle.randn(shape, dtype=dtype)

class RandomProjector:
    """Random projection for gradients."""
    def __init__(
        self, 
        rank, 
        shape=None, 
        update_proj_gap=1, 
        scale=1.0, 
        proj_type='std', 
        seed=0
    ):
        self.rank = rank
        self.shape = shape
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.proj_type = proj_type
        self.seed = seed
        self.ortho_matrix = None
        self.svd_count = 0
        
    def update_ortho_matrix(self, full_rank_grad, proj_type):
        """
        Updates the orthogonal matrix based on the projection type.

        Args:
            full_rank_grad: The full rank gradient matrix.
            proj_type: Projection type ('left', 'right').
        """
        self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type=proj_type, seed=self.seed)
        
    def project(self, full_rank_grad, iter):
        """
        Projects the gradient to a lower rank.

        Args:
            full_rank_grad: The full rank gradient matrix.
            iter: Current iteration number.

        Returns:
            The projected low-rank gradient.
        """
        # 确保梯度是2D张量
        if len(full_rank_grad.shape) != 2:
            raise ValueError(f"Gradient must be 2D tensor, got shape {full_rank_grad.shape}")
            
            
        if self.proj_type == 'std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                # 高矩阵，使用右投影
                if self.ortho_matrix is None or iter % max(1, int(1/self.update_proj_gap)) == 0:
                    self.update_ortho_matrix(full_rank_grad, proj_type='right')
                # 右投影：[m,n] x [r,n].T -> [m,r]
                low_rank_grad = paddle.matmul(full_rank_grad, paddle.transpose(self.ortho_matrix, [1, 0]))
            else:
                # 宽矩阵，使用左投影
                if self.ortho_matrix is None or iter % max(1, int(1/self.update_proj_gap)) == 0:
                    self.update_ortho_matrix(full_rank_grad, proj_type='left')
                # 左投影：[m,r].T x [m,n] -> [r,n]
                low_rank_grad = paddle.matmul(paddle.transpose(self.ortho_matrix, [1, 0]), full_rank_grad)
        elif self.proj_type == 'reverse_std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                # 高矩阵，但使用左投影
                if self.ortho_matrix is None or iter % max(1, int(1/self.update_proj_gap)) == 0:
                    self.update_ortho_matrix(full_rank_grad, proj_type='left')
                # 左投影：[m,r].T x [m,n] -> [r,n]
                low_rank_grad = paddle.matmul(paddle.transpose(self.ortho_matrix, [1, 0]), full_rank_grad)
            else:
                # 宽矩阵，但使用右投影
                if self.ortho_matrix is None or iter % max(1, int(1/self.update_proj_gap)) == 0:
                    self.update_ortho_matrix(full_rank_grad, proj_type='right')
                # 右投影：[m,n] x [r,n].T -> [m,r]
                low_rank_grad = paddle.matmul(full_rank_grad, paddle.transpose(self.ortho_matrix, [1, 0]))
        elif self.proj_type == 'right':
            # 强制使用右投影
            if self.ortho_matrix is None or iter % max(1, int(1/self.update_proj_gap)) == 0:
                self.update_ortho_matrix(full_rank_grad, proj_type='right')
            # 右投影：[m,n] x [r,n].T -> [m,r]
            low_rank_grad = paddle.matmul(full_rank_grad, paddle.transpose(self.ortho_matrix, [1, 0]))
        elif self.proj_type == 'left':
            # 强制使用左投影
            if self.ortho_matrix is None or iter % max(1, int(1/self.update_proj_gap)) == 0:
                self.update_ortho_matrix(full_rank_grad, proj_type='left')
            # 左投影：[m,r].T x [m,n] -> [r,n]
            low_rank_grad = paddle.matmul(paddle.transpose(self.ortho_matrix, [1, 0]), full_rank_grad)
        elif self.proj_type == 'full':
            raise NotImplementedError("full rank projection is not implemented yet")

            
        return low_rank_grad
        
    def project_back(self, low_rank_grad):
        """
        Projects the low-rank gradient back to the original space.

        Args:
            low_rank_grad: The low-rank gradient.

        Returns:
            The full-rank gradient.
        """
            
        if self.proj_type == 'std':
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                # [m,r] x [r,n] -> [m,n]
                full_rank_grad = paddle.matmul(low_rank_grad, self.ortho_matrix)
            else:
                # [r,n] x [m,r] -> [m,n]
                full_rank_grad = paddle.matmul(self.ortho_matrix, low_rank_grad)
        elif self.proj_type == 'reverse_std':
            if low_rank_grad.shape[0] <= low_rank_grad.shape[1]:  # note this is different from std
                # [r,n] x [m,r] -> [m,n]
                full_rank_grad = paddle.matmul(self.ortho_matrix, low_rank_grad)
            else:
                # [m,r] x [r,n] -> [m,n]
                full_rank_grad = paddle.matmul(low_rank_grad, self.ortho_matrix)
        elif self.proj_type == 'right':
            # [m,r] x [r,n] -> [m,n]
            full_rank_grad = paddle.matmul(low_rank_grad, self.ortho_matrix)
        elif self.proj_type == 'left':
            # [r,n] x [m,r] -> [m,n]
            full_rank_grad = paddle.matmul(self.ortho_matrix, low_rank_grad)
        elif self.proj_type == 'full':
            raise NotImplementedError("full rank projection is not implemented yet")
            
            
        return full_rank_grad * self.scale
        
    def get_orthogonal_matrix(self, weights, rank, type, seed):
        """
        Generates an orthogonal projection matrix.

        Args:
            weights: Tensor to determine the shape of the projection matrix.
            rank: Target rank for the projection.
            type: Type of projection ('left', 'right').
            seed: Seed for generating the matrix.

        Returns:
            The generated orthogonal matrix.
        """
        module_params = weights
        float_data = module_params.dtype == paddle.float32
        original_type = module_params.dtype
        matrix = module_params.astype(paddle.float32) if not float_data else module_params

        # Generate projection matrix in a variance of sqrt(1/r)
        if type == "left":
            proj = stable_randn(
                [matrix.shape[0], rank], seed=seed, dtype=matrix.dtype
            ) / np.sqrt(rank)
        elif type == "right":
            proj = stable_randn(
                [rank, matrix.shape[1]], seed=seed, dtype=matrix.dtype
            ) / np.sqrt(rank)
        elif type == "full":
            raise NotImplementedError("full rank projection is not implemented yet")
        else:
            raise ValueError("type should be left, right or full")

        if not float_data:
            proj = proj.astype(original_type)
        return proj

class SVDProjector:
    """SVD-based projection for gradients."""
    def __init__(self, rank, shape=None, update_proj_gap=1, verbose=False, scale=1.0, proj_type='std'):
        """
        Initializes the SVDProjector.

        Args:
            rank: Target rank for the projection.
            shape: Shape of the parameter (optional).
            update_proj_gap: Iterations before updating the orthogonal matrix.
            verbose: If True, print additional information.
            scale: Scaling factor for the projection.
            proj_type: Type of projection ('std', 'reverse_std', 'left', 'right', 'full').
        """
        self.rank = rank
        self.shape = shape
        self.update_proj_gap = update_proj_gap
        self.verbose = verbose
        self.scale = scale
        self.proj_type = proj_type
        self.ortho_matrix = None
        self.svd_count = 0
        
    def project(self, full_rank_grad, iter):
        """
        Projects the gradient to a lower rank.

        Args:
            full_rank_grad: The full rank gradient matrix.
            iter: Current iteration number.

        Returns:
            The projected low-rank gradient.
        """
        # 确保梯度是2D张量
        if len(full_rank_grad.shape) != 2:
            raise ValueError(f"Gradient must be 2D tensor, got shape {full_rank_grad.shape}")
            
        if self.verbose and iter % 100 == 0:
            print(f"SVDProjector: Input grad shape: {full_rank_grad.shape}, rank: {self.rank}")
            
        # Check if ortho_matrix needs to be moved to the same device as the gradient
        if self.ortho_matrix is not None and hasattr(self.ortho_matrix, 'place') and hasattr(full_rank_grad, 'place'):
            if str(self.ortho_matrix.place) != str(full_rank_grad.place):
                if isinstance(self.ortho_matrix, list):
                    self.ortho_matrix = [m.cuda() if 'gpu' in str(full_rank_grad.place) else m.cpu() for m in self.ortho_matrix]
                else:
                    self.ortho_matrix = self.ortho_matrix.cuda() if 'gpu' in str(full_rank_grad.place) else self.ortho_matrix.cpu()

        if self.proj_type == 'std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                # 高矩阵，使用右投影
                if self.ortho_matrix is None or iter % max(1, int(1/self.update_proj_gap)) == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                    self.svd_count += 1
                # 右投影：[m,n] x [r,n].T -> [m,r]
                low_rank_grad = paddle.matmul(full_rank_grad, paddle.transpose(self.ortho_matrix, [1, 0]))
            else:
                # 宽矩阵，使用左投影
                if self.ortho_matrix is None or iter % max(1, int(1/self.update_proj_gap)) == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                    self.svd_count += 1
                # 左投影：[m,r].T x [m,n] -> [r,n]
                low_rank_grad = paddle.matmul(paddle.transpose(self.ortho_matrix, [1, 0]), full_rank_grad)
        elif self.proj_type == 'reverse_std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                # 高矩阵，但使用左投影
                if self.ortho_matrix is None or iter % max(1, int(1/self.update_proj_gap)) == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                    self.svd_count += 1
                # 左投影：[m,r].T x [m,n] -> [r,n]
                low_rank_grad = paddle.matmul(paddle.transpose(self.ortho_matrix, [1, 0]), full_rank_grad)
            else:
                # 宽矩阵，但使用右投影
                if self.ortho_matrix is None or iter % max(1, int(1/self.update_proj_gap)) == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                    self.svd_count += 1
                # 右投影：[m,n] x [r,n].T -> [m,r]
                low_rank_grad = paddle.matmul(full_rank_grad, paddle.transpose(self.ortho_matrix, [1, 0]))
        elif self.proj_type == 'right':
            # 强制使用右投影
            if self.ortho_matrix is None or iter % max(1, int(1/self.update_proj_gap)) == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                self.svd_count += 1
            # 右投影：[m,n] x [r,n].T -> [m,r]
            low_rank_grad = paddle.matmul(full_rank_grad, paddle.transpose(self.ortho_matrix, [1, 0]))
        elif self.proj_type == 'left':
            # 强制使用左投影
            if self.ortho_matrix is None or iter % max(1, int(1/self.update_proj_gap)) == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                self.svd_count += 1
            # 左投影：[m,r].T x [m,n] -> [r,n]
            low_rank_grad = paddle.matmul(paddle.transpose(self.ortho_matrix, [1, 0]), full_rank_grad)
        elif self.proj_type == 'full':
            # 同时使用左右投影
            if self.ortho_matrix is None or iter % max(1, int(1/self.update_proj_gap)) == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='full')
                self.svd_count += 1
            # 全投影：[m,r].T x [m,n] x [r,n].T -> [r,r]
            low_rank_grad = paddle.matmul(
                paddle.matmul(paddle.transpose(self.ortho_matrix[0], [1, 0]), full_rank_grad),
                paddle.transpose(self.ortho_matrix[1], [1, 0])
            )
                
        if self.verbose and iter % 100 == 0:
            print(f"SVDProjector: Output grad shape: {low_rank_grad.shape}")
            
        return low_rank_grad

    def project_back(self, low_rank_grad):
        """
        Projects the low-rank gradient back to the original space.

        Args:
            low_rank_grad: The low-rank gradient.

        Returns:
            The full-rank gradient.
        """
        if self.verbose:
            print(f"SVDProjector: Project back input shape: {low_rank_grad.shape}")
            
        if self.proj_type == 'std':
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                # [m,r] x [r,n] -> [m,n]
                full_rank_grad = paddle.matmul(low_rank_grad, self.ortho_matrix)
            else:
                # [r,n] x [m,r] -> [m,n]
                full_rank_grad = paddle.matmul(self.ortho_matrix, low_rank_grad)
        elif self.proj_type == 'reverse_std':
            if low_rank_grad.shape[0] <= low_rank_grad.shape[1]:  # note this is different from std
                # [r,n] x [m,r] -> [m,n]
                full_rank_grad = paddle.matmul(self.ortho_matrix, low_rank_grad)
            else:
                # [m,r] x [r,n] -> [m,n]
                full_rank_grad = paddle.matmul(low_rank_grad, self.ortho_matrix)
        elif self.proj_type == 'right':
            # [m,r] x [r,n] -> [m,n]
            full_rank_grad = paddle.matmul(low_rank_grad, self.ortho_matrix)
        elif self.proj_type == 'left':
            # [r,n] x [m,r] -> [m,n]
            full_rank_grad = paddle.matmul(self.ortho_matrix, low_rank_grad)
        elif self.proj_type == 'full':
            # [m,r] x [r,r] x [r,n] -> [m,n]
            full_rank_grad = paddle.matmul(
                paddle.matmul(self.ortho_matrix[0], low_rank_grad),
                self.ortho_matrix[1]
            )
            
        if self.verbose:
            print(f"SVDProjector: Project back output shape: {full_rank_grad.shape}")
            
        return full_rank_grad * self.scale

    def get_orthogonal_matrix(self, weights, rank, type):
        """
        Generates an orthogonal projection matrix using SVD.

        Args:
            weights: Tensor to determine the shape of the projection matrix.
            rank: Target rank for the projection.
            type: Type of projection ('left', 'right', 'full').

        Returns:
            The generated orthogonal matrix or matrices.
        """
        module_params = weights
        float_data = module_params.dtype == paddle.float32
        original_type = module_params.dtype
        matrix = module_params.astype(paddle.float32) if not float_data else module_params

        U, s, Vh = paddle.linalg.svd(matrix, full_matrices=False)

        if type == 'right':
            A = paddle.matmul(U[:, :rank], paddle.diag(s[:rank]))
            B = Vh[:rank, :]
            if not float_data:
                B = B.astype(original_type)
            return B
        elif type == 'left':
            A = U[:, :rank]
            B = paddle.matmul(paddle.diag(s[:rank]), Vh[:rank, :])
            if not float_data:
                A = A.astype(original_type)
            return A
        elif type == 'full':
            A = U[:, :rank]
            B = Vh[:rank, :]
            if not float_data:
                A = A.astype(original_type)
                B = B.astype(original_type)
            return [A, B]
        else:
            raise ValueError("type should be 'left', 'right', or 'full'")
