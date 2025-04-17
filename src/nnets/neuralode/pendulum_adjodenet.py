import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from typing import Iterator
from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint

from ..mlp import MLPConditionalGenerator
from src.sampler.distributions import sample_latent_noise


class NeuralPendulum(nn.Module):

    def __init__(
        self,
        nnet: nn.Module,
        physics_model: nn.Module,
        t_intg: torch.Tensor,
        cfg_model,
        device=None,
        nnet_time_embedding: callable = None,
    ):
        """
        Parameters
        ----------
        nnet: nn.Module
            The neural network to use of the ODE

        ode_solver: str
            The solver to use for the ODE. Options are 'dopri5', 'adams', 'bdf', 'rk4', 'midpoint', 'euler'
        t_intg: torch.Tensor
            1-D Tensor containing the evaluation points, it corresponds to the time points to integrate over.
        """
        super(NeuralPendulum, self).__init__()
        self.device = device

        self.nnet = nnet
        self.z_dim = nnet.z_dim
        self.physics_model = physics_model
        self.c_dim = nnet.c_dim
        self.params_dim = nnet.c_dim
        self.cfg_model = cfg_model
        self.z_std = 0.1 if self.z_dim > 0 else 0.0
        self.nnet_time_embedding = nnet_time_embedding
        self.nnet_f_evals = []
        self.full_f_evals = []
        self.count_f_evals = 0
        self.use_time_dim = (
            True
            if "time_dim_flag" not in cfg_model or cfg_model["time_dim_flag"]
            else False
        )

        self.with_phys = (
            False if cfg_model["phys_model"].lower() != "pendulum" else True
        )
        self.ode_solver = (
            cfg_model["ode_solver"] if "ode_solver" in cfg_model else "euler"
        )
        self.ode_stepsize = (
            cfg_model["ode_stepsize"] if "ode_stepsize" in cfg_model else None
        )
        self.t_intg = t_intg
        self.step_size = (t_intg[-1] - t_intg[0]) / (len(t_intg) - 1)
        self.start_t = t_intg[0]
        self.len_intg = len(t_intg)
        self.adjoint_flag = (
            cfg_model["adjoint_solver"]
            if "adjoint_solver" in cfg_model
            else False
        )

        self.cfg_model = cfg_model

        self.nnet_aux = None
        if "aux_conf" in cfg_model:
            for key, value in cfg_model["aux_conf"].items():
                if "f_aux" in key:
                    if value["enable"]:
                        h_layers = value["h_layers"]
                        act = value["activation"]
                        if not value["in_loop_ode"]:
                            in_dim = self.t_intg.shape[
                                0
                            ]  # trajectory dimension
                        else:
                            in_dim = 1
                        
                        self.nnet_aux = MLPConditionalGenerator(
                            in_dim,
                            c_dim=self.nnet.c_dim,
                            z_dim=in_dim if self.z_dim > 0 else 0,
                            output_dim=(
                                self.nnet.x_dim
                                if value["in_loop_ode"]
                                else in_dim
                            ),
                            h_layers=h_layers,
                            activation=act,
                            device=device,
                        )

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """
        Returns an iterator over module parameters.
        This is typically passed to an optimizer.
        """
        params = self.nnet.parameters(recurse=recurse)

        if self.nnet_aux is not None:
            params = list(params)
            params += self.nnet_aux.parameters(recurse=recurse)

        return params

    def forward(self, t, state):
        yy_PA = torch.atleast_2d(
            state
        )  # Black box neural ode not supported yet. Should use flag with_phys.
        if self.with_phys:
            yy_dot_phy_PA = self.physics_model(
                init_conds=yy_PA, params=self.phys_params
            )
        else:
            yy_dot_phy_PA = 0.0

        ode_context = self.phys_params

        if self.z_dim > 0:
            # index_t = (int)((t - self.start_t) / self.step_size)
            if self.z_dim == 1:
                z = self.x[
                    :, -1:
                ]  # one noise sample for the entire trajectory
            else:
                z = self.x[
                    :, -self.z_dim :
                ]  # noise sample for each state-time point

            # [x(t), alpha(t), t, z]
            if self.use_time_dim:
                x_input = torch.cat(
                    [yy_PA, t.expand(yy_PA.shape[0], 1), z], dim=1
                )
            else:
                x_input = torch.cat([yy_PA, z], dim=1)  # yy_PA
        else:
            if self.nnet_time_embedding is not None:
                time_emb = self.nnet_time_embedding(t)
                x_input = torch.cat(
                    [yy_PA, time_emb.expand(yy_PA.shape[0], 1)], dim=1
                )
            else:
                if self.use_time_dim:
                    x_input = torch.cat(
                        [yy_PA, t.expand(yy_PA.shape[0], 1)], dim=1
                    )
                else:
                    x_input = yy_PA

        nnet_f = self.nnet(x_input, context=ode_context)
        self.nnet_f_evals.append(
            torch.cat([nnet_f, t.expand(nnet_f.shape[0], 1)], dim=1)
        )
        yy_dot_aux_PA = torch.cat(
            [
                torch.zeros(yy_PA.shape[0], 1, device=self.device),
                nnet_f,
            ],
            dim=1,
        )

        new_grad_state = torch.cat([yy_dot_phy_PA + yy_dot_aux_PA], dim=1)
        self.full_f_evals.append(new_grad_state)
        return new_grad_state

    def simulate(self, x: torch.Tensor, context: torch.Tensor):
        """
        Parameters
        ----------
        """
        self.x = x  # only used for retrieving the noise samples

        # Take the physics parameters from the context, not need the initial.
        if context.shape[1] > 1:
            self.phys_params = context[:, : self.params_dim].reshape(
                -1, self.params_dim
            )
        else:
            self.phys_params = context

        self.nnet_f_evals = []
        self.full_f_evals = []
        # Extract the initial conditions from the input x. The initial conditions are given by the input x.
        
        init_x = x[:, 0].reshape(-1, 1)
        n_samples = self.phys_params.shape[0]
        alpha = torch.zeros(n_samples, 1, device=self.device)
        init_cond = torch.cat([init_x, alpha], dim=1)  # <n x 2>

        if self.adjoint_flag:
            yy_seq = odeint_adjoint(
                self,
                init_cond,
                self.t_intg,
                method=self.ode_solver,
                options={"step_size": self.ode_stepsize},
            )

        else:
            yy_seq = odeint(
                self,
                init_cond,
                self.t_intg,
                method=self.ode_solver,
                options={"step_size": self.ode_stepsize},
            )  # <len_intg x n x 2or4>

        Y = yy_seq[:, :, 0].T

        
        if self.nnet_aux is not None:
            if self.z_dim > 0:
                z = x[:, self.t_intg.shape[0] :]
                X_input = torch.cat([Y, z], dim=1)
            else:
                X_input = Y

            
            Y = Y + self.nnet_aux(X_input, context=self.phys_params)

        return Y  # self.nnet.variance

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
            self.nnet_f_evals = []
            return self.simulate(X, context=context)

        x_dim = X.shape[-1]
        # Repeat for noise samples
        XZ = X.reshape(-1, 1, x_dim).repeat(1, z_samples, 1)

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
        XZ = torch.cat([XZ, Z], dim=2)

        # Prediction
        XZ = XZ.flatten(start_dim=0, end_dim=1)
        params = params.flatten(start_dim=0, end_dim=1)
        pred = self.simulate(XZ, context=params)
        return pred.reshape(X.shape[0], z_samples, -1)
