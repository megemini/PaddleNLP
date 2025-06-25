# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import warnings
import math
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING

import paddle
from paddle.optimizer import AdamW

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paddle import Tensor

__all__ = []


class GradientProjector:
    """
    A class to project gradients to a lower rank using random orthogonal matrices.
    """
    def __init__(self, rank: int, verbose: bool = False, update_proj_gap: float = 0.01,
                 scale: float = 1.0, proj_type: str = 'std', proj: str = 'random',
                 scale_type: str = 'tensor', seed: int = 0):
        """
        Initializes the GradientProjector.

        Args:
            rank (int): The rank of the projection.
            verbose (bool): Whether to print verbose information.
            update_proj_gap (float): The gap between projection updates.
            scale (float): The scale factor.
            proj_type (str): The type of projection ('std', 'reverse_std', 'right', 'left', 'full').
            proj (str): The projection method ('random' or 'svd').
            scale_type (str): The type of scaling ('tensor' or 'channel').
            seed (int): Random seed for projection matrix generation.
        """
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.proj_type = proj_type
        self.proj = proj
        self.scale_type = scale_type
        self.ortho_matrix = None
        self.svd_count = 0
        self.seed = seed

    def project(self, full_rank_grad: paddle.Tensor, iter: int) -> paddle.Tensor:
        """
        Projects the full-rank gradient to a lower rank.

        Args:
            full_rank_grad (paddle.Tensor): The full-rank gradient.
            iter (int): The current iteration.

        Returns:
            paddle.Tensor: The projected low-rank gradient.
        """
        # Store original shape for project_back
        self._original_shape = full_rank_grad.shape

        if self.proj == 'random':
            return self._project_random(full_rank_grad, iter)
        elif self.proj == 'svd':
            return self._project_svd(full_rank_grad, iter)
        else:
            raise ValueError(f"Unknown projection method: {self.proj}")

    def project_back(self, low_rank_grad: paddle.Tensor) -> paddle.Tensor:
        """
        Projects the low-rank gradient back to the original space.

        Args:
            low_rank_grad (paddle.Tensor): The low-rank gradient.

        Returns:
            paddle.Tensor: The full-rank gradient.
        """
        if self.proj_type == 'std':
            # For std projection, we need to reverse the projection
            # If original was tall and we projected from right: low_rank @ ortho_matrix.T
            # If original was wide and we projected from left: ortho_matrix.T @ low_rank
            if hasattr(self, '_original_shape'):
                if self._original_shape[0] >= self._original_shape[1]:
                    # Was tall matrix, projected from right
                    full_rank_grad = paddle.matmul(low_rank_grad, self.ortho_matrix.T)
                else:
                    # Was wide matrix, projected from left
                    full_rank_grad = paddle.matmul(self.ortho_matrix.T, low_rank_grad)
            else:
                # Fallback: assume right projection
                full_rank_grad = paddle.matmul(low_rank_grad, self.ortho_matrix.T)
        elif self.proj_type == 'reverse_std':
            if hasattr(self, '_original_shape'):
                if self._original_shape[0] >= self._original_shape[1]:
                    # Was tall matrix, projected from left
                    full_rank_grad = paddle.matmul(self.ortho_matrix.T, low_rank_grad)
                else:
                    # Was wide matrix, projected from right
                    full_rank_grad = paddle.matmul(low_rank_grad, self.ortho_matrix.T)
            else:
                # Fallback: assume left projection
                full_rank_grad = paddle.matmul(self.ortho_matrix.T, low_rank_grad)
        elif self.proj_type == 'right':
            full_rank_grad = paddle.matmul(low_rank_grad, self.ortho_matrix.T)
        elif self.proj_type == 'left':
            full_rank_grad = paddle.matmul(self.ortho_matrix.T, low_rank_grad)
        elif self.proj_type == 'full':
            raise NotImplementedError("full rank projection is not implemented yet")
        else:
            raise ValueError(f"Unknown projection type: {self.proj_type}")

        return full_rank_grad

    def _project_random(self, full_rank_grad: paddle.Tensor, iter: int) -> paddle.Tensor:
        """Random projection implementation"""
        # Handle update_proj_gap like torch version
        update_gap = int(self.update_proj_gap) if self.update_proj_gap >= 1 else max(1, int(1/self.update_proj_gap))

        if self.proj_type == "std":
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                # Tall matrix: project from right (columns)
                if self.ortho_matrix is None or iter % update_gap == 0:
                    self._update_ortho_matrix_random(full_rank_grad, proj_type="right")
                # full_rank_grad: [m, n], ortho_matrix: [n, rank] -> result: [m, rank]
                low_rank_grad = paddle.matmul(full_rank_grad, self.ortho_matrix.T)
            else:
                # Wide matrix: project from left (rows)
                if self.ortho_matrix is None or iter % update_gap == 0:
                    self._update_ortho_matrix_random(full_rank_grad, proj_type="left")
                # ortho_matrix: [rank, m], full_rank_grad: [m, n] -> result: [rank, n]
                low_rank_grad = paddle.matmul(self.ortho_matrix.T, full_rank_grad)
        elif self.proj_type == "reverse_std":
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                # Tall matrix: project from left (rows)
                if self.ortho_matrix is None or iter % update_gap == 0:
                    self._update_ortho_matrix_random(full_rank_grad, proj_type="left")
                low_rank_grad = paddle.matmul(self.ortho_matrix.T, full_rank_grad)
            else:
                # Wide matrix: project from right (columns)
                if self.ortho_matrix is None or iter % update_gap == 0:
                    self._update_ortho_matrix_random(full_rank_grad, proj_type="right")
                low_rank_grad = paddle.matmul(full_rank_grad, self.ortho_matrix.T)
        elif self.proj_type == "right":
            if self.ortho_matrix is None or iter % update_gap == 0:
                self._update_ortho_matrix_random(full_rank_grad, proj_type="right")
            low_rank_grad = paddle.matmul(full_rank_grad, self.ortho_matrix.T)
        elif self.proj_type == "left":
            if self.ortho_matrix is None or iter % update_gap == 0:
                self._update_ortho_matrix_random(full_rank_grad, proj_type="left")
            low_rank_grad = paddle.matmul(self.ortho_matrix.T, full_rank_grad)
        elif self.proj_type == "full":
            raise NotImplementedError("full rank projection is not implemented yet")
        else:
            raise ValueError("type should be std, reverse_std, right, left or full")

        return low_rank_grad

    def _project_svd(self, full_rank_grad: paddle.Tensor, iter: int) -> paddle.Tensor:
        """SVD projection implementation"""
        if self.proj_type == 'std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % max(1, int(1/self.update_proj_gap)) == 0:
                    self.ortho_matrix = self._get_orthogonal_matrix_svd(full_rank_grad, self.rank, type='right')
                    self.svd_count += 1
                low_rank_grad = paddle.matmul(full_rank_grad, self.ortho_matrix.T)
            else:
                if self.ortho_matrix is None or iter % max(1, int(1/self.update_proj_gap)) == 0:
                    self.ortho_matrix = self._get_orthogonal_matrix_svd(full_rank_grad, self.rank, type='left')
                    self.svd_count += 1
                low_rank_grad = paddle.matmul(self.ortho_matrix.T, full_rank_grad)
        elif self.proj_type == 'reverse_std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % max(1, int(1/self.update_proj_gap)) == 0:
                    self.ortho_matrix = self._get_orthogonal_matrix_svd(full_rank_grad, self.rank, type='left')
                    self.svd_count += 1
                low_rank_grad = paddle.matmul(self.ortho_matrix.T, full_rank_grad)
            else:
                if self.ortho_matrix is None or iter % max(1, int(1/self.update_proj_gap)) == 0:
                    self.ortho_matrix = self._get_orthogonal_matrix_svd(full_rank_grad, self.rank, type='right')
                    self.svd_count += 1
                low_rank_grad = paddle.matmul(full_rank_grad, self.ortho_matrix.T)
        elif self.proj_type == 'right':
            if self.ortho_matrix is None or iter % max(1, int(1/self.update_proj_gap)) == 0:
                self.ortho_matrix = self._get_orthogonal_matrix_svd(full_rank_grad, self.rank, type='right')
                self.svd_count += 1
            low_rank_grad = paddle.matmul(full_rank_grad, self.ortho_matrix.T)
        elif self.proj_type == 'left':
            if self.ortho_matrix is None or iter % max(1, int(1/self.update_proj_gap)) == 0:
                self.ortho_matrix = self._get_orthogonal_matrix_svd(full_rank_grad, self.rank, type='left')
                self.svd_count += 1
            low_rank_grad = paddle.matmul(self.ortho_matrix.T, full_rank_grad)
        elif self.proj_type == 'full':
            raise NotImplementedError("full rank projection is not implemented yet")
        else:
            raise ValueError("type should be std, reverse_std, right, left or full")

        return low_rank_grad

    def _update_ortho_matrix_random(self, matrix: paddle.Tensor, proj_type: str):
        """Update orthogonal matrix using random projection"""
        # Set random seed for reproducible results
        paddle.seed(self.seed)

        if proj_type == "left":
            # For left projection: [matrix.shape[0], rank]
            proj = paddle.randn([matrix.shape[0], self.rank], dtype=matrix.dtype) / math.sqrt(self.rank)
        elif proj_type == "right":
            # For right projection: [rank, matrix.shape[1]]
            proj = paddle.randn([self.rank, matrix.shape[1]], dtype=matrix.dtype) / math.sqrt(self.rank)
        else:
            raise ValueError("proj_type should be left or right")

        self.ortho_matrix = proj
        # Update seed for next time (similar to torch version)
        self.seed = (self.seed + 1) % 2**32

    def _get_orthogonal_matrix_svd(self, matrix: paddle.Tensor, rank: int, type: str) -> paddle.Tensor:
        """Get orthogonal matrix using SVD"""
        if type == 'left':
            U, _, _ = paddle.linalg.svd(matrix, full_matrices=False)
            return U[:, :rank].T
        elif type == 'right':
            _, _, Vh = paddle.linalg.svd(matrix, full_matrices=False)
            return Vh[:rank, :]
        else:
            raise ValueError("type should be left or right")


class Apollo(AdamW):
    r"""
    The Apollo optimizer is implemented based on the Apollo Optimization algorithm.
    It uses low-rank projections to reduce memory usage while maintaining optimization performance.

    Args:
        learning_rate (float|LRScheduler, optional): The learning rate used to update ``Parameter``.
            It can be a float value or a LRScheduler. The default value is 0.001.
        beta1 (float|Tensor, optional): The exponential decay rate for the 1st moment estimates.
            It should be a float number or a 0-D Tensor with shape [] and data type as float32.
            The default value is 0.9.
        beta2 (float|Tensor, optional): The exponential decay rate for the 2nd moment estimates.
            It should be a float number or a 0-D Tensor with shape [] and data type as float32.
            The default value is 0.999.
        epsilon (float|Tensor, optional): A small float value for numerical stability.
            The default value is 1e-08.
        parameters (list|tuple|None, optional): List/Tuple of ``Tensor`` names to update to minimize ``loss``.
            This parameter is required in dygraph mode. And you can specify different options for
            different parameter groups such as the learning rate, weight decay, etc,
            then the parameters are list of dict. Note that the learning_rate in parameter groups
            represents the scale of base learning_rate.
            The default value is None in static graph mode, at this time all parameters will be updated.
        weight_decay (int|float|Tensor, optional): The weight decay coefficient, it can be int, float or Tensor. The default value is 0.01.
        scale_front (bool, optional): Whether to scale the gradient before projection. Default is False.
        grad_clip (GradientClipBase|None, optional): Gradient clipping strategy, it's an instance of
            some derived class of ``GradientClipBase`` . There are three clipping strategies
            ( :ref:`api_paddle_nn_ClipGradByGlobalNorm` , :ref:`api_paddle_nn_ClipGradByNorm` ,
            :ref:`api_paddle_nn_ClipGradByValue` ). Default None, meaning there is no gradient clipping.
        name (str|None, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`.
            The default value is None.
    """

    helper: None
    type: str
    _moment1_acc_str = "moment1"
    _moment2_acc_str = "moment2"
    _beta1_pow_acc_str = "beta1_pow_acc"
    _beta2_pow_acc_str = "beta2_pow_acc"

    def __init__(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        parameters=None,
        weight_decay=0.01,
        scale_front=False,
        grad_clip=None,
        name=None,
    ):
        # Initialize parent AdamW
        super().__init__(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            parameters=parameters,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            name=name,
        )

        # Apollo-specific attributes
        self._projectors = {}
        self._step_count = 0
        self._scale_front = scale_front
        self._param_seeds = {}

        # Assign seeds to parameters (similar to torch version)
        params_idx = 0
        if isinstance(self._param_groups[0], dict):
            for group in self._param_groups:
                for p in group['params']:
                    params_idx += 1
                    self._param_seeds[p.name] = params_idx

    def _initialize_projector(self, group, param):
        """Initialize projector for a parameter group"""
        seed = self._param_seeds.get(param.name, 0)
        if group.get('proj', 'random') == 'random':
            projector = GradientProjector(
                rank=group['rank'],
                update_proj_gap=group.get('update_proj_gap', 0.01),
                scale=group.get('scale', 1.0),
                proj_type=group.get('proj_type', 'std'),
                proj='random',
                scale_type=group.get('scale_type', 'tensor'),
                seed=seed
            )
        else:  # SVD
            projector = GradientProjector(
                rank=group['rank'],
                update_proj_gap=group.get('update_proj_gap', 0.01),
                scale=group.get('scale', 1.0),
                proj_type=group.get('proj_type', 'std'),
                proj='svd',
                scale_type=group.get('scale_type', 'tensor'),
                seed=seed
            )
        return projector

    def step(self):
        """Override step method to handle Apollo projection"""
        if not isinstance(self._param_groups[0], dict):
            # No parameter groups, use standard AdamW
            return super().step()

        # Handle parameter groups with Apollo projection
        for group in self._param_groups:
            params_with_grad = []
            grads = []

            for p in group['params']:
                if p.stop_gradient:
                    continue
                if p._grad_ivar() is not None:
                    params_with_grad.append(p)
                    grads.append(p._grad_ivar())

            if not params_with_grad:
                continue

            # Apply Apollo projection if this group has rank
            if "rank" in group:
                self._apollo_step(params_with_grad, grads, group)
            else:
                # For standard parameters, call parent step method
                # Create a temporary optimizer with just these parameters
                temp_params = [{'params': params_with_grad}]
                temp_optimizer = AdamW(
                    parameters=temp_params,
                    learning_rate=self._learning_rate,
                    beta1=self._beta1,
                    beta2=self._beta2,
                    epsilon=self._epsilon,
                    weight_decay=self._weight_decay
                )
                temp_optimizer.step()

    def _apollo_step(self, params, grads, group):
        """Apply Apollo optimization step"""
        self._step_count += 1

        for param, grad in zip(params, grads):
            if len(grad.shape) <= 1:
                # Skip 1D parameters (like bias)
                continue

            # Initialize projector if not exists
            if param.name not in self._projectors:
                self._projectors[param.name] = self._initialize_projector(group, param)

            projector = self._projectors[param.name]

            # APOLLO Step 1: Project gradient to low-rank space
            projected_grad = projector.project(grad, self._step_count)

            # Get or create accumulators for this parameter
            if param.name not in self._accumulators[self._moment1_acc_str]:
                # Create accumulators with projected shape
                proj_shape = projected_grad.shape
                self._accumulators[self._moment1_acc_str][param.name] = paddle.zeros(proj_shape, dtype=param.dtype)
                self._accumulators[self._moment2_acc_str][param.name] = paddle.zeros(proj_shape, dtype=param.dtype)
                self._accumulators[self._beta1_pow_acc_str][param.name] = paddle.to_tensor([self._beta1], dtype=param.dtype)
                self._accumulators[self._beta2_pow_acc_str][param.name] = paddle.to_tensor([self._beta2], dtype=param.dtype)
                self._accumulators['step'][param.name] = paddle.to_tensor([0], dtype='int64')

            # Get accumulators
            moment1 = self._accumulators[self._moment1_acc_str][param.name]
            moment2 = self._accumulators[self._moment2_acc_str][param.name]
            beta1_pow = self._accumulators[self._beta1_pow_acc_str][param.name]
            beta2_pow = self._accumulators[self._beta2_pow_acc_str][param.name]
            step_count = self._accumulators['step'][param.name]

            step_count += 1

            # APOLLO Step 2: Update moments in projected space
            moment1 *= self._beta1
            moment1 += (1.0 - self._beta1) * projected_grad

            moment2 *= self._beta2
            moment2 += (1.0 - self._beta2) * projected_grad * projected_grad

            # Compute step in projected space
            denom = moment2.sqrt() + self._epsilon
            norm_grad_proj = moment1 / denom

            # Apply bias correction
            step_size = self.get_lr()
            bias_correction1 = 1.0 - beta1_pow
            bias_correction2 = 1.0 - beta2_pow
            step_size = step_size * paddle.sqrt(bias_correction2) / bias_correction1

            norm_grad_proj *= step_size

            # APOLLO Step 3: Calculate gradient scaling factor
            norm_dim = 0 if grad.shape[0] < grad.shape[1] else 1

            if group.get('scale_type', 'tensor') == 'channel':
                # Channel-wise scaling
                grad_scaling_factor = (
                    paddle.norm(norm_grad_proj, axis=norm_dim) /
                    (paddle.norm(projected_grad, axis=norm_dim) + 1e-8)
                )
                if norm_dim == 1:
                    grad_scaling_factor = grad_scaling_factor.unsqueeze(1)
            else:
                # Tensor-wise scaling
                grad_scaling_factor = (
                    paddle.norm(norm_grad_proj) /
                    (paddle.norm(projected_grad) + 1e-8)
                )

            # APOLLO Step 4: Scale original gradient
            scaled_grad = grad * grad_scaling_factor

            if self._scale_front:
                scaled_grad *= math.sqrt(group.get('scale', 1.0))

            norm_grad = scaled_grad

            if not self._scale_front:
                norm_grad *= math.sqrt(group.get('scale', 1.0))

            # Apply parameter update
            param -= norm_grad

            # Apply weight decay
            if self._weight_decay > 0:
                param -= self.get_lr() * self._weight_decay * param

            # Update beta powers
            beta1_pow *= self._beta1
            beta2_pow *= self._beta2

    def __str__(self):
        return "Apollo Optimizer"
