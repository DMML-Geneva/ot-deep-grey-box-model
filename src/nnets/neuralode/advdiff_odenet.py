from typing import Iterator, Tuple
from torch.nn.parameter import Parameter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchdiffeq import odeint, odeint_adjoint

from src.forward_models.advdiff import AdvDiffDynamics
from src.sampler.distributions import sample_latent_noise


class NeuralAdvDiff(nn.Module):

    def __init__(
        self,
        dx: float,
        dt: float,
        len_episode,
        n_grids,
        nnet: nn.Module,
        cfg_model: dict,
        store_time_eval=False,
        device="cuda",
    ):

        super(NeuralAdvDiff, self).__init__()
        self.dt = dt
        self.dx = dx
        self.len_episode = len_episode
        self.t_intg = torch.linspace(
            0.0, dt * (len_episode - 1), len_episode
        ).to(device)
        self.n_grids = (n_grids,)
        self.store_time_eval = store_time_eval
        self.method = (
            cfg_model["ode_solver"] if "ode_solver" in cfg_model else "euler"
        )
        self.physics_prior_flag = (
            False if cfg_model["phys_model"].lower() != "advdiff" else True
        )
        self.device = device
        self.nnet = nnet
        self.z_dim = nnet.z_dim
        self.params_dim = 1

        self.cfg_model = cfg_model
        self.adjoint_flag = (
            bool(cfg_model["adjoint_solver"])
            if "adjoint_solver" in cfg_model
            else False
        )
        self.z_std = 0.1 if self.z_dim > 0 else 0.0

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """
        Returns an iterator over module parameters.
        This is typically passed to an optimizer.

        Returns
        -------
        Iterator[Parameter]
            an iterator over module parameters
        """
        params = self.nnet.parameters(recurse=recurse)
        return params

    def forward(self, t, state):
        """
        Forward model of advection-diffusion equation.
        ----------
        Args:

            context: Physiscs Parameters of the model. Diffusion coefficient (dcoeff) and convection coefficient (ccoeff).
            x: Initial condition.
        Returns:
        """
        if self.physics_prior_flag:
            phys_step_function = AdvDiffDynamics(
                params=self.context, dx=self.dx, device=self.device
            )
            phys_f_eval = phys_step_function(t, state)
        else:
            phys_f_eval = 0.0

        n_samples = state.shape[0]
        # state n_samples \times n_grids
        if self.z_dim > 0:
            if self.z_dim == 1:
                z = self.x[
                    :, 0, -1:
                ]  # one noise sample for the entire trajectory, n_samples x 1. All the `n_grids` boundaries share the same noise sample.
            else:
                z = self.x[
                    :, 0, -self.z_dim :
                ]  # n_samples x z_dim. Still one noise sample for the entire trajectory.

        context = torch.cat(
            [
                self.context[:, 0, :],
                torch.sin(t).expand(n_samples, 1),
                torch.cos(t).expand(n_samples, 1),
            ],
            dim=-1,
        )
        if self.z_dim > 0:
            context = torch.cat([context, z], dim=-1)

        nnet_eval = self.nnet(state, context=context)
        self.nnet_f_evals.append(nnet_eval)
        new_grad_state = phys_f_eval + nnet_eval
        self.full_f_evals.append(new_grad_state)
        return new_grad_state

    def simulate(self, x: torch.Tensor, context: torch.Tensor):
        self.x = x  # only used for retrieving the noise samples

        self.context = context
        self.nnet_f_evals = []
        self.full_f_evals = []

        # The neural ode only requires the initial state/condition t=0
        init_cond = x[:, :, 0]
        if self.adjoint_flag:
            ode_sol = odeint_adjoint(
                self, init_cond, self.t_intg, method=self.method
            )
        else:
            ode_sol = odeint(self, init_cond, self.t_intg, method=self.method)

        # extract to <n x dim_y x dim_t>
        Y = ode_sol.permute(1, 2, 0).contiguous()

        return Y

    def predict(
        self,
        X,
        context=None,
        z_samples=10,
        z_type_dist="gauss",
        z_std=0.1,
        len_episode=None,
    ):
        # change model's integrator setting
        old_len_episode = self.len_episode
        if len_episode is not None:
            self.len_episode = len_episode
            self.t_intg = torch.linspace(
                0.0,
                self.dt * self.len_episode,
                self.len_episode,
                device=self.device,
            )

        if self.z_dim < 1:
            self.nnet_f_evals = []
            self.full_f_evals = []
            return self.simulate(X, context=context)

        x_dim = X.shape[1:]
        # Repeat for noise samples
        sims = X.reshape((-1, 1) + (x_dim)).repeat(1, z_samples, 1, 1)
        params = context.reshape(-1, 1, x_dim[0], context.shape[-1]).repeat(
            1, z_samples, 1, 1
        )

        Z = sample_latent_noise(
            n_samples=X.shape[0],
            z_dim=self.z_dim,
            z_size=z_samples,
            type=z_type_dist,
            z_std=z_std,
            device=X.device,
        )

        Z = Z.reshape(-1, z_samples, 1, self.z_dim).repeat(
            (1, 1) + x_dim[:-1] + (1,)
        )

        XZ = torch.cat([sims, Z], dim=-1)

        # Prediction
        pred = self.simulate(
            XZ.flatten(start_dim=0, end_dim=1),
            context=params.flatten(start_dim=0, end_dim=1),
        )

        # reset len_episode
        if len_episode is not None:
            self.len_episode = old_len_episode

        return pred


class AdvDiffNNet(nn.Module):

    def __init__(
        self,
        state_dim,
        z_dim=1,
        c_dim=0,
        output_dim=None,
        conv_nfilters=4,
        kernel_size=3,
        h_layers=[64, 64],  # 64 neurons in each hidden layer
        activation="relu",
        device="cuda",
    ):
        super(AdvDiffNNet, self).__init__()
        self.state_dim = state_dim
        self.h_layers = h_layers
        self.z_dim = z_dim
        self.c_dim = self.z_dim + c_dim

        self.output_dim = output_dim

        self.activation_str = activation
        if self.activation_str == "relu":
            self.activation = F.relu
        elif self.activation_str == "leaky_relu":
            self.activation = F.leaky_relu
        elif self.activation_str == "elu":
            self.activation = F.elu
        elif self.activation_str == "selu":
            self.activation = F.selu
        elif self.activation_str == "tanh":
            self.activation = F.tanh
        elif self.activation_str == "softplus":
            # Suggest by FAQ Odenet of Chen et al. 2018.
            # Prefer non-linearities with a theoretically unique adjoint/gradient such as Softplus
            # https://github.com/rtqichen/torchdiffeq/blob/cae73789b929d4dbe8ce955dace0089cf981c1a0/FAQ.md
            self.activation = F.softplus
        elif self.activation_str == "sine":
            self.activation = lambda x: torch.sin(x)

        self.device = device
        self.conv_nfilters = conv_nfilters
        self.kernel_size = kernel_size

        # First convolutional layer
        self.conv_input = nn.Conv1d(
            1,
            1 * self.conv_nfilters,
            kernel_size=kernel_size,
            padding=1,
            stride=1,
        )
        if self.c_dim > 0:
            self.fc_context_input = nn.Linear(self.c_dim, self.conv_nfilters)

        from src.nnets.utils import get_conv1d_output_dim

        conv1_out_dim = get_conv1d_output_dim(
            self.state_dim, kernel_size=kernel_size, stride=1, padding=1
        )

        # First fully connected layer and context block
        self.fc1 = nn.Linear(conv1_out_dim * self.conv_nfilters, h_layers[0])
        self.context_blocks = [nn.Linear(self.c_dim, h_layers[0])]

        # Hidden blocks
        self.h_blocks = []

        for i in range(1, len(h_layers)):
            self.h_blocks.append(nn.Linear(h_layers[i - 1], h_layers[i]))
            if c_dim > 0:
                self.context_blocks.append(nn.Linear(self.c_dim, h_layers[i]))

        self.h_blocks = nn.ModuleList(self.h_blocks)
        self.context_blocks = nn.ModuleList(self.context_blocks)

        # Output layer
        self.output_layer = nn.Linear(h_layers[-1], self.output_dim)

        self.to(device)

        from src.train.ot_physics.init_nnet_task import weights_init_nnet

        apply_weight = lambda m: weights_init_nnet(
            m, activation=self.activation_str
        )
        self.apply(apply_weight)

    def forward(self, x, context=None):

        nsamples = x.shape[0]
        out = x.view(nsamples, 1, -1)  # n_samples x 1 channel x n_grids

        # first convolution
        out = self.activation(self.conv_input(out))
        out = out.view(nsamples, -1)
        # first linear layer
        out = self.activation(self.fc1(out))
        # first context layer
        out = self.activation(self.context_blocks[0](context) + out)

        for i, block in enumerate(self.h_blocks):
            out = block(out)
            out = self.activation(out)
            # Conditional block
            if context is not None and i < len(self.context_blocks):
                c = self.context_blocks[i + 1](context)
                out = out + self.activation(c)

        out = self.output_layer(out)
        return out
