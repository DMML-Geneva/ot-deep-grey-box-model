"""Generate data of forced damped pendulum.
"""

import torch
from torch import nn
from torchdiffeq import odeint
import numpy as np

import argparse
import json
import time

from src.forward_models.pendulum.visualization import plot_phase_space


class PendulumSolver(nn.Module):
    def __init__(
        self,
        len_episode,
        dt=0.01,
        noise_loc=None,
        noise_std=None,
        with_grad=False,
        store_time_eval=True,
        method="rk4",
    ):
        super().__init__()
        self.model_type = "pendulum"
        self.len_episode = len_episode
        self.noise_loc = noise_loc
        self.noise_std = noise_std
        self.with_grad = with_grad
        self.store_time_eval = store_time_eval
        self.method = method
        if isinstance(self.method, list):
            self.method = self.method[0]
        self.dt = dt
        self.dt_friction = []
        self.dt_states = []

    def forward(self, init_conds, params: torch.Tensor, t_grad=False):
        """
        Forward computation of the forced damped pendulum.
        ----------
        Args:
            params: torch.tensor
                Parameters of the model. The first element is the natural frequency omega, the second element is the damping gamma, the third element is the amplitude A, and the fourth element is the phase phi.
            init_conds: torch.tensor
                Initial condition of the pendulum. The first element is the angle theta, the second element is the angular velocity theta_dot.
        Returns:
            x: torch.tensor
                The observation of the pendulum. The first element is the angle theta, the second element is the angular velocity theta_dot.
            sol: torch.tensor
                The solution of the pendulum. The first element is the angle theta, the second element is the angular velocity theta_dot.
            t: torch.tensor
                The time steps of the solution.
            time_eval: float
                The time it took to evaluate the solution.
        """
        params = torch.atleast_2d(params)
        init_conds = torch.atleast_2d(init_conds)

        omega = params[:, 0].view(-1, 1)
        gamma = (
            params[:, 1].view(-1, 1)
            if params.shape[1] > 1
            else torch.zeros_like(omega)
        )
        A = (
            params[:, 2].view(-1, 1)
            if params.shape[1] > 2
            else torch.zeros_like(omega)
        )
        phi = (
            params[:, 3].view(-1, 1)
            if params.shape[1] > 3
            else torch.zeros_like(omega)
        )
        t = torch.linspace(
            0.0,
            self.dt * (self.len_episode - 1),
            self.len_episode,
            device=omega.device,
        )

        # reinitialize dt_friction for each simulation run (forwards)
        self.dt_friction = []
        self.dt_states = []

        def fun(t, s):
            th, thdot = s[:, 0].view(-1, 1), s[:, 1].view(-1, 1)
            force = (
                A * omega * omega * torch.cos(2.0 * torch.pi * phi * t)
            )  # damping force
            dt_friction = gamma * thdot
            self.dt_states.append(
                torch.cat([th, thdot, t.expand(th.shape[0], 1)], dim=1)
            )
            self.dt_friction.append(
                torch.cat(
                    [dt_friction, t.expand(dt_friction.shape[0], 1)], dim=1
                )
            )
            return torch.cat(
                [
                    thdot,
                    force - dt_friction - omega * omega * torch.sin(th),
                ],
                dim=-1,
            )

        start_t = time.perf_counter() if self.store_time_eval else 0.0
        if not self.with_grad:
            with torch.no_grad():
                sol = odeint(fun, init_conds, t, method=self.method)
        else:
            sol = odeint(fun, init_conds, t, method=self.method)
        end_t = time.perf_counter() if self.store_time_eval else 0.0

        assert (
            sol.shape[-1] == 2
        ), "Initial value solution should be 2D, first dim is time, second dim is state"
        assert sol.shape[0] == (
            self.len_episode
        ), "Time dimension should be equal to len_episode"

        x = sol[:, :, 0]
        # permute x
        x = x.permute(1, 0)

        # observation noise
        if self.noise_loc is None and self.noise_std is not None:
            self.noise_loc = 0.0
        if (
            self.noise_loc is not None
            and self.noise_std is not None
            and self.noise_std > 0.0
        ):
            x = x + torch.randn_like(x) * self.noise_std + self.noise_loc

        return {
            "x": x,
            "sol": sol,
            "t": t,
            "time_eval": (
                (end_t - start_t) + 1e-9 if self.store_time_eval else 0.0
            ),
        }


class SimplePendulum(nn.Module):
    """
    Simple pendulum model with no damping and no forcing, used for the OdeNet model.
    This class does not perform the odeint integration, it only returns the derivative of the state.
    """

    def __init__(
        self,
    ):
        super(SimplePendulum, self).__init__()
        self.n_init_conds = (2,)
        self.n_params = 1

    def forward(self, init_conds: torch.Tensor, params: torch.Tensor):
        """
        given parameter and yy=[y, dy/dt], return dyy/dt=[dy/dt, d^2y/dt^2]
        [state]
            yy: shape <n x 2>
        [physics parameter]
            omega_sq: shape <n x 1>
        ----------
        Returns:
            dy/dt: torch.tensor
                The derivative of the state.
            d^2y/dt^2: torch.tensor
                The second derivative of the state.
        """
        return torch.cat(
            [
                init_conds[:, 1].reshape(-1, 1),
                -(params**2) * torch.sin(init_conds[:, 0].view(-1, 1)),
            ],
            dim=1,
        )
