import torch
from torch import nn
import numpy as np


"""
From https://github.com/yiftachbeer/mmd_loss_pytorch 
Implementation of MMD Loss in Pytorch, forked from ZongxianLee/MMD_Loss.Pytorch. 
"""


class RBF(nn.Module):

    def __init__(
        self, n_kernels=5, mul_factor=2.0, bandwidth=None, device="cpu"
    ):
        super().__init__()
        self.bandwidth = (
            bandwidth.to(device) if bandwidth is not None else None
        )
        self.device = device

        self.bandwidth_multipliers = (
            mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        ).to(device)

    @staticmethod
    def estimate_bandwidth(X, median_heuristic=False):
        n_samples = X.shape[0]
        L2_distances = torch.cdist(X, X) ** 2
        if median_heuristic:
            bandwidth = torch.sqrt(L2_distances[L2_distances != 0].median())
        else:
            bandwidth = L2_distances.data.sum() / (n_samples**2 - n_samples)

        del L2_distances
        return bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples**2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(
            -L2_distances[None, ...]
            / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[
                :, None, None
            ]
        ).sum(dim=0)


"""
From https://github.com/yiftachbeer/mmd_loss_pytorch 
Implementation of MMD Loss in Pytorch, forked from ZongxianLee/MMD_Loss.Pytorch. 

This class uses different RBF kernel with different bandwidths and aggregates them, by summing them up.
"""


class MMDLoss(nn.Module):

    def __init__(self, kernel=None):
        super().__init__()
        self.kernel = kernel
        self.device = kernel.device
        if self.kernel is None:
            self.kernel = RBF()

    def forward(self, X, Y, max_batch_size=None):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY


def estimate_mmd_bandwidth(
    val_y_sampler, median_heuristic=True, joint_samples=False, params_dim=1
):
    bandwidth = []
    for idx in range(len(val_y_sampler)):
        samples = val_y_sampler.sample()
        X = samples["x"]
        if X.dim() > 2:
            # n_samples x noisy_samples x dim
            X_dim = np.prod(X.shape[2:])
            X = X.reshape(-1, X_dim)

        if joint_samples:
            params = samples["params"].flatten(start_dim=0, end_dim=1)
            if params.ndim > 2:
                params = params[:, :, :params_dim]
            else:
                params = params[:, :params_dim]
            params = params.reshape(-1, np.prod(params.shape[1:]))
            Y = torch.cat([X, params], dim=-1)

        rbf_length_scaler = RBF.estimate_bandwidth(
            X, median_heuristic=median_heuristic
        )

        bandwidth.append(rbf_length_scaler)

    bandwidth = torch.tensor(bandwidth, device=val_y_sampler.device).mean()
    return bandwidth
