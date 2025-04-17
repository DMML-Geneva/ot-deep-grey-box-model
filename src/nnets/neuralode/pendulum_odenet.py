import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from typing import Iterator
from torchdiffeq import odeint
from src.sampler.distributions import sample_latent_noise

from ..mlp import MLPConditionalGenerator

from logging import getLogger
import time


class NeuralPendulum(nn.Module):
    def __init__(
        self,
        mlp: nn.Module,
        physics_model: nn.Module,
        ode_solver: str,
        t_intg: torch.Tensor,
        cfg_model,
        device=None,
        nnet_time_embedding: callable = None,
    ):
        """
        Parameters
        ----------
        mlp: nn.Module
            The neural network that will be used to model the ODE.
        physics_model: nn.Module
            The physics model to use.
        ode_solver: str
            The ODE solver to use.
        t_intg: torch.Tensor
            The time points to integrate the ODE.
        cfg_model: dict
            The configuration of the model.
        device: str
            The device to use.

        """
        super(NeuralPendulum, self).__init__()

        self.logger = getLogger("ADV-KNOT")
        self.mlp = mlp
        self.physics_model = physics_model
        self.ode_solver = ode_solver
        self.t_intg = t_intg
        self.output_dim = len(t_intg)
        self.step_size = (t_intg[-1] - t_intg[0]) / (len(t_intg) - 1)
        self.start_t = t_intg[0]
        self.z_dim = mlp.z_dim
        self.c_dim = mlp.c_dim
        self.params_dim = mlp.c_dim
        self.device = device
        self.with_phys = (
            False if cfg_model["phys_model"].lower() != "pendulum" else True
        )
        self.cfg_model = cfg_model
        self.z_std = 0.1 if self.z_dim > 0 else 0.0
        self.use_time_dim = (
            True
            if "time_dim_flag" not in cfg_model or cfg_model["time_dim_flag"]
            else False
        )
        self.nnet_time_embedding = nnet_time_embedding
        self.nnet_f_evals = []
        self.full_f_evals = []
        self.count_f_evals = 0

        if not self.with_phys:
            self.logger.warn(
                "No physics prior model provided. Using black-box ODE."
            )

        self.nnet_aux = None
        # Auxiliary MLP that takes the output of the ODE and add/map the output space.
        if "aux_conf" in cfg_model:
            for key, value in cfg_model["aux_conf"].items():
                if "f_aux" in key:
                    if value["enable"]:
                        h_layers = value["h_layers"]
                        act = value["activation"]
                        in_dim = 1  # initial condition dimension

                        self.nnet_aux = MLPConditionalGenerator(
                            in_dim,
                            c_dim=self.mlp.c_dim,
                            z_dim=in_dim if self.z_dim > 0 else 0,
                            output_dim=(
                                self.mlp.x_dim
                                if value["in_loop_ode"]
                                else self.output_dim
                            ),
                            h_layers=h_layers,
                            activation=act,
                            device=device,
                        )

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """
        Returns an iterator over module parameters.
        This is typically passed to an optimizer.

        Returns
        -------
        Iterator[Parameter]
            an iterator over module parameters
        """
        params = self.mlp.parameters(recurse=recurse)

        if self.nnet_aux is not None:
            params = list(params)
            params += self.nnet_aux.parameters(recurse=recurse)

        return params

    def simulate(self, x: torch.Tensor, context: torch.Tensor, t_grad=False):
        return self(x, context, t_grad=t_grad)

    def forward(self, x: torch.Tensor, context: torch.Tensor, t_grad=False):
        """
        Parameters
        ----------
        x: torch.Tensor
            The input tensor of shape <n x m>.

        context: torch.Tensor
            Physics parameters of shape <n x p>
        """
        # Only take the physics parameters from the context, not need the initial.
        # The initial conditions are given by the input x.
        if context.shape[1] > 1:
            phys_params = context[:, : self.params_dim].reshape(
                -1, self.params_dim
            )
        else:
            phys_params = context
        n = phys_params.shape[0]
        self.nnet_f_evals = []
        self.full_f_evals = []

        def ODEfunc(t: torch.Tensor, _yy: torch.Tensor):
            """

            Gives gradient of vector _yy, whose shape is <n x 4> or <n x 2>.
            - t should be a scalar
            - _yy should be shape <n x 2>.
            """
            yy_PA = _yy[
                :, [0, 1]
            ]  # <n x 2>  # [x(t), alpha(t)]  # x(t) = current state, alpha(t) = current velocity
            if self.with_phys:
                # physics part (omega & yy --> time-deriv of yy)
                yy_dot_phy_PA = self.physics_model(
                    init_conds=yy_PA, params=phys_params
                )
            else:
                # when model has no physics part *originally*, black box ODENet
                yy_dot_phy_PA = 0.0

            ode_context = phys_params

            if self.z_dim > 0:
                # index_t = (int)((t - self.start_t) / self.step_size)
                if self.z_dim == 1:
                    z = x[:, -1:]  # one noise sample for the entire trajectory
                else:
                    z = x[
                        :, -self.z_dim :
                    ]  # z_dim noise samples for the entire trajectory

                # [x(t), alpha(t), t, z]
                if self.use_time_dim:
                    x_input = torch.cat(
                        [yy_PA, t.expand(yy_PA.shape[0], 1), z], dim=1
                    )
                else:
                    x_input = torch.cat([yy_PA, z], dim=1)
            else:
                if self.use_time_dim:
                    x_input = torch.cat([yy_PA, t.expand(n, 1)], dim=1)
                else:
                    x_input = (yy_PA,)

            f_nnet = self.mlp(x_input, context=ode_context)
            self.nnet_f_evals.append(f_nnet)
            yy_dot_aux_PA = torch.cat(
                [
                    torch.zeros(n, 1, device=self.device),
                    f_nnet,
                ],
                dim=1,
            )
            new_grad_state = torch.cat([yy_dot_phy_PA + yy_dot_aux_PA], dim=1)
            self.full_f_evals.append(new_grad_state)
            return new_grad_state

        # Solve ODE
        # Extract the initial conditions from the input x
        init_x = x[:, 0].reshape(-1, 1)
        tmp = torch.zeros(n, 1, device=self.device)
        init_cond = torch.cat([init_x, tmp], dim=1)  # <n x 2>
        if t_grad:
            self.t_intg.requires_grad_(True)

        yy_seq = odeint(
            ODEfunc, init_cond, self.t_intg, method=self.ode_solver
        )

        self.count_f_evals = len(self.nnet_f_evals)
        y_seq_PA = yy_seq[:, :, 0].T

        if t_grad:
            self.time_grad = torch.autograd.grad(
                y_seq_PA,
                self.t_intg,
                grad_outputs=torch.ones(
                    y_seq_PA.shape, device=y_seq_PA.device
                ),
                create_graph=True,
            )[0]

        x_PA = y_seq_PA.clone()
        
        if self.nnet_aux is not None:
            if self.z_dim > 0:
                if self.z_dim == 1:
                    z = x[:, -1:]
                else:
                    z = x[:, self.t_intg.shape[0] :]
                    z = z[:, : init_x.shape[-1]].reshape(-1, 1)
                X_input = torch.cat([init_x, z], dim=1)
            else:
                X_input = init_x

            
            #       f_odenet + f_aux(params, zaux1, zaux2) = trajectory
            x_PA = x_PA + self.nnet_aux(X_input, phys_params)

        return x_PA  # self.mlp.variance

    def predict(
        self,
        X,
        context=None,
        z_samples=10,
        z_type_dist="gauss",
        z_std=0.1,
        t_grad=False,
    ):
        if self.z_dim < 1:
            return self.simulate(X, context=context)

        x_dim = X.shape[-1]
        # Repeat for noise samples
        sims = X.reshape(-1, 1, x_dim).repeat(1, z_samples, 1)
        params = context.reshape(-1, 1, context.shape[-1]).repeat(
            1, z_samples, 1
        )

        Z = sample_latent_noise(
            n_samples=X.shape[0],
            z_dim=self.z_dim,
            z_size=z_samples,
            type=z_type_dist,
            z_std=z_std,
            device=X.device,
        )

        XZ = torch.cat([sims, Z], dim=2)

        # Prediction
        XZ = XZ.flatten(start_dim=0, end_dim=1)
        params = params.flatten(start_dim=0, end_dim=1)
        pred = self.simulate(XZ, context=params, t_grad=t_grad)
        return pred.reshape(-1, z_samples, x_dim)

    def extrapolate(self, x, T_horizon, dt):
        x = torch.atleast_2d(x)
        tr_dt = self.step_size
        # n_traj_chunks = f_evals / self.len_intg
        # better to think in periods of the trajectory
        # tr_period = tr_dt * self.len_intg
        # test_dt = dt
