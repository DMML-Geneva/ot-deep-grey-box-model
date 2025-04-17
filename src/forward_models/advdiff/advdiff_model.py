import torch
from torch import nn

from torch.nn import functional as F
from torchdiffeq import odeint

import time

HASHED_KEYS_NAME = "hashed_keys"

HASHED_KEYS = [
    "dt",
    "dx",
    "n_grids",
    "len_episode",
    "n_samples",
    "n_stoch_samples",
    "range_init_mag",
    "range_dcoeff",
    "range_ccoeff",
    "noise_loc",
    "noise_std",
    "seed",
    "method",
    "library",
]


class AdvDiffDynamics(nn.Module):
    def __init__(
        self,
        params: torch.Tensor,
        dx: float,
        device="cuda",
    ):
        super(AdvDiffDynamics, self).__init__()
        self.dx = dx
        params = torch.atleast_2d(params)
        # get the last dimension of the params any generic dimension of the params
        if params.ndim == 3:  # params as n_samples x n_grid x dim_params
            params = params.reshape(-1, params.shape[-1])

        self.dcoeff = params[:, 0].view(-1, 1)
        self.ccoeff = (
            params[:, -1].view(-1, 1) if params.shape[-1] > 1 else None
        )

        # Discrete Laplacian
        self.register_buffer(
            "discLap1",
            torch.Tensor([-1.0, 0.0, 1.0])
            .to(device=device)
            .unsqueeze(0)
            .unsqueeze(0),  # 1 x 1 x 3
        )
        self.register_buffer(
            "discLap2",
            torch.Tensor([1.0, -2.0, 1.0])
            .to(device=device)
            .unsqueeze(0)
            .unsqueeze(0),
        )
        self.discLap2.requires_grad = False
        self.discLap1.requires_grad = False

    def forward(self, t, init_conds: torch.Tensor):
        """
        Step function of the advection-diffusion equation, used in the ODE solver.
        ----------
        Args:
            t: Time points.
            init_conds: Initial condition.

        Returns:
            res: Solution of the advection-diffusion equation.
        """
        y_xx = F.conv1d(
            init_conds.unsqueeze(1),
            weight=self.discLap2,
            stride=1,
            padding=1,
            bias=None,
        ).squeeze(1)

        res = self.dcoeff.view_as(y_xx) * y_xx / self.dx / self.dx

        if self.ccoeff is not None:
            y_x = F.conv1d(
                init_conds.unsqueeze(1),
                weight=self.discLap1,
                stride=1,
                padding=1,
                bias=None,
            ).squeeze(1)
            res = res - self.ccoeff.view_as(y_xx) * y_x / 2.0 / self.dx

        return res


class AdvDiffSolver(nn.Module):

    def __init__(
        self,
        dx: float,
        dt: float,
        len_episode,
        n_grids,
        noise_loc=None,
        noise_std=None,
        store_time_eval=True,
        method="euler",
        device="cuda",
    ):
        super(AdvDiffSolver, self).__init__()
        self.dx = dx
        self.dt = dt
        self.len_episode = len_episode
        self.n_grids = n_grids

        # Time and space grids
        self.t = torch.linspace(
            0.0,
            self.dt * (self.len_episode - 1),
            self.len_episode,
            device=device,
        )
        self.x_grid = torch.linspace(
            0.0, self.dx * (self.n_grids - 1), self.n_grids, device=device
        )
        self.device = device

        # noise
        self.noise_loc = noise_loc
        self.noise_std = noise_std
        self.store_time_eval = store_time_eval
        self.method = method

    def forward(
        self,
        init_conds: torch.Tensor,
        params: torch.tensor,
    ):
        """
        Forward model of advection-diffusion equation.
        ----------
        Args:

            params: Parameters of the model. Diffusion coefficient (dcoeff) and convection coefficient (ccoeff).
            y: Initial condition.
        Returns:
            t: Time points.
            x: Noisy solution.
            ode_sol: Clean solution.
            time: Time taken to solve the ODE.
        """
        params = torch.atleast_2d(params)
        init_conds = torch.atleast_2d(init_conds)
        n = params.shape[0]

        start_t = time.perf_counter() if self.store_time_eval else 0.0
        phys_step_function = AdvDiffDynamics(
            params, self.dx, device=self.device
        )

        # ODE solver
        ode_sol = odeint(
            phys_step_function, init_conds, self.t, method=self.method
        )

        # extract to <n x dim_y x dim_t>
        x = ode_sol.permute(1, 2, 0).contiguous()
        end_t = time.perf_counter() if self.store_time_eval else 0.0

        # observation noise
        if self.noise_loc is None and self.noise_std is not None:
            self.noise_loc = 0.0
        if self.noise_loc is not None and self.noise_std is not None:
            x = x + torch.randn_like(x) * self.noise_std + self.noise_loc

        return {
            "x": x,
            "sol": ode_sol,
            "init_conds": init_conds,
            "t": self.t,
            "x_grid": self.x_grid,
            "time_eval": (
                (end_t - start_t) + 1e-9 if self.store_time_eval else 0.0
            ),
        }
