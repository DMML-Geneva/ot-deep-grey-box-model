from typing import Iterator, Tuple
from torch.nn.parameter import Parameter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchdiffeq import odeint, odeint_adjoint

from src.forward_models.reactdiff.reactdiff_model import ReactDiffDynamics
from src.sampler.distributions import sample_latent_noise


class NeuralReactDiff(nn.Module):

    def __init__(
        self,
        t0: float,
        t1: float,
        dx: float,
        len_episode,
        cfg_model: dict,
        z_dim=4,
        store_time_eval=False,
        device="cuda",
    ):

        super(NeuralReactDiff, self).__init__()
        self.t0 = t0
        self.t1 = t1
        self.dx = dx
        self.len_episode = len_episode
        self.state_dim = 2
        # Time and space grids
        self.t_intg = torch.linspace(
            self.t0,
            self.t1,
            self.len_episode,
            device=device,
        )
        self.react_diff_model = ReactDiffDynamics(
            params=None, dx=self.dx, device=device
        )
        self.grid_size = (
            cfg_model["grid_size"] if "grid_size" in cfg_model else 32
        )
        self.physics_prior_flag = (
            False if cfg_model["phys_model"].lower() != "reactdiff" else True
        )
        self.solver_cfg = cfg_model["ode_solver_cfg"]
        self.method = self.solver_cfg["method"]

        # NNet model
        self.det_model = z_dim > 0
        self.z_dim = z_dim
        self.params_dim = 2
        self.nnet = ReactDiffNNet(
            z_dim=z_dim, c_dim=self.params_dim, device=device
        )
        self.device = device

        self.cfg_model = cfg_model
        self.adjoint_flag = (
            bool(cfg_model["adjoint_solver"])
            if "adjoint_solver" in cfg_model
            else True
        )
        self.z_std = 0.1 if self.z_dim > 0 else 0.0
        self.store_time_eval = store_time_eval
        self.to(device)

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
        Forward pass of the neural ODE model.
        Parameters
        ----------
        t: torch.Tensor
            Time points.
        state: torch.Tensor
            State of the system.
        """
        n_samples = state.shape[0]
        if self.physics_prior_flag:
            self.react_diff_model.params = self.context
            phys_f_eval = self.react_diff_model(t, state)
        else:
            phys_f_eval = 0.0

        # get z noise samples
        if self.z_dim > 0:
            # n_samples x 2 x grid x grid x (time + z) ->
            if self.z_dim == 1:
                z = self.x[:, :, :, :, -1:]
            else:
                z = self.x[:, :, :, :, -self.z_dim :]
                z = z[:, 0]
            z = z.permute(0, 3, 1, 2)  # n_samples x z_dim x grid x grid

            state_input = torch.cat([state, z], dim=1)
        else:
            state_input = state
        context = self.context
        

        nnet_eval = self.nnet(state_input, context=context)
        self.nnet_f_evals.append(nnet_eval)

        dfdt = phys_f_eval + nnet_eval

        self.full_f_evals.append(dfdt)
        return dfdt

    def simulate(self, x: torch.Tensor, context: torch.Tensor):
        self.x = x

        self.context = context
        self.nnet_f_evals = []
        self.full_f_evals = []

        # The neural ode only requires the initial state/condition at t=0
        init_cond = x[
            :, : self.state_dim, :, :, 0
        ]  # n_samples x 2 x grid x grid x time
        if self.adjoint_flag:
            ode_sol = odeint_adjoint(
                self, init_cond, self.t_intg, **self.solver_cfg
            )
        else:
            ode_sol = odeint(self, init_cond, self.t_intg, **self.solver_cfg)

        Y = ode_sol.permute(
            1, 2, 3, 4, 0
        )  # n_samples, 2, grid_size, grid_size, time (len_episode)

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

        if self.z_dim < 1:
            self.nnet_f_evals = []
            self.full_f_evals = []
            return self.simulate(X, context=context)

        # Repeat for noise samples

        Z = sample_latent_noise(
            n_samples=X.shape[0],
            z_dim=self.z_dim,
            z_size=z_samples,
            type=z_type_dist,
            z_std=z_std,
            device=X.device,
        )

        sims = X.unsqueeze(1).repeat(
            1, z_samples, 1, 1, 1, 1
        )  # n_samples x z_size x 2 x grid x grid x time

        # Z shape from (n_samples, z_size, z_dim) to (n_samples, z_size, state, grid, grid, z_dim)
        Z = (
            Z.unsqueeze(2)
            .unsqueeze(2)
            .unsqueeze(2)
            .repeat(1, 1, self.state_dim, self.grid_size, self.grid_size, 1)
        )
        # [(n_samples x z_size x 2 x grid x grid x time), (n_samples x z_size x 2 x grid x grid x z_dim)]
        XZ = torch.cat([sims, Z], dim=-1)
        XZ = XZ.flatten(
            start_dim=0, end_dim=1
        )  # (n_samples * z_size) x 2 x grid x grid x (time + z_dim)
        params = (
            context.unsqueeze(1)
            .repeat(1, z_samples, 1)
            .flatten(start_dim=0, end_dim=1)
        )  # n_samples x z_size x c_dim

        # Prediction
        pred = self.simulate(
            XZ,
            context=params,
        )

        return pred


class ReactDiffNNet(nn.Module):

    def __init__(
        self,
        z_dim=4,
        c_dim=0,
        h_layers=[1],  # 64 neurons in each hidden layer
        h_ch_dim=16,  # 16 channels or 32
        grid_size=2**5,
        c_emb_dim=2**10,
        det_c_emb_dim=0,  # 2**5
        activation="leaky_relu",
        device="cuda",
    ):
        super(ReactDiffNNet, self).__init__()
        self.state_dim = 2
        self.grid_size = grid_size
        self.h_layers = h_layers
        self.h_ch_dim = h_ch_dim if h_ch_dim > 0 else int(self.grid_size // 2)

        self.z_dim = z_dim
        self.c_dim = c_dim

        # self.output_dim = state_dim * grid_size * grid_size

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
            self.activation = F.softplus
        elif self.activation_str == "sine":
            self.activation = lambda x: torch.sin(x)

        self.device = device
        # Convolutional input layer
        self.conv_input = nn.Conv2d(
            self.state_dim + z_dim,  # input channels
            self.h_ch_dim,  # output channels
            3,
            stride=1,
            padding=1,
        )  # kernel size and padding

        hidden_layers = [
            nn.Conv2d(self.h_ch_dim, self.h_ch_dim, 3, stride=1, padding=1)
        ]
        self.h_blocks = nn.ModuleList(hidden_layers)

        self.output_layer = nn.Conv2d(self.h_ch_dim, 2, 3, stride=1, padding=1)

        # Context Layers
        self.context_blocks = None

        if self.c_dim > 0:
            self.det_c_emb_dim = det_c_emb_dim
            if self.det_c_emb_dim > 0:
                omega = np.pi * torch.arange(1, det_c_emb_dim + 1)
                # bufferize omega
                self.register_buffer(
                    "_omega",
                    torch.tensor(omega, device=device, dtype=torch.float32),
                )

                def omega_fn(c):
                    return torch.cat(
                        [
                            torch.sin(self._omega * c),
                            torch.cos(self._omega * c),
                        ],
                        dim=-1,
                    )

                self.bathched_emb_vmap = torch.vmap(omega_fn)

                self.c_dim = det_c_emb_dim * 2
            else:
                self.bathched_emb_vmap = None

            self.c_emb_dim = c_emb_dim
            self.context_layer = [
                nn.Linear(self.c_dim, c_emb_dim)
                for _ in range(len(hidden_layers) + 1)
            ]
            self.context_blocks = nn.ModuleList(self.context_layer)

        self.to(device)

        from src.train.ot_physics.init_nnet_task import weights_init_nnet

        apply_weight = lambda m: weights_init_nnet(
            m, activation=self.activation_str
        )
        self.apply(apply_weight)

    def forward(self, x, context=None):
        """
        Forward pass of the neural ODE model.
        Parameters
        ----------
        x: torch.Tensor
            Input tensor. Input shape n \times 2+z_dim \times \grid_size \times \grid_size
        context: torch.Tensor
            Context tensor. Input shape n \times c_dim
        """

        nsamples = x.shape[0]

        out = x.view(
            nsamples,
            self.state_dim + self.z_dim,
            self.grid_size,
            self.grid_size,
        )
        # x->out.shape: n_samples x 2+z_dim x grid x grid

        # first convolution
        out = self.activation(self.conv_input(out))
        # out.shape: n_samples x h_out x grid x grid

        # X context layer
        if context is not None:
            if self.det_c_emb_dim > 0 and self.bathched_emb_vmap is not None:
                c = self.bathched_emb_vmap(context)
            else:
                c = context
            c = self.context_blocks[0](c)
            c = (
                c.reshape(nsamples, self.grid_size, self.grid_size)
                .unsqueeze(1)
                .expand(*out.shape)
            )
            out = out + self.activation(c)

        for i, block in enumerate(self.h_blocks):
            out = block(out)
            out = self.activation(out)
            if context is not None:
                if (
                    self.det_c_emb_dim > 0
                    and self.bathched_emb_vmap is not None
                ):
                    c = self.bathched_emb_vmap(context)
                else:
                    c = context
                c = self.context_blocks[i + 1](c)
                c = (
                    c.reshape(nsamples, self.grid_size, self.grid_size)
                    .unsqueeze(1)
                    .expand(*out.shape)
                )
                out = out + self.activation(c)

        out = self.output_layer(out)
        return out


if __name__ == "__main__":

    z_dim = 2
    c_dim = 2
    nnet = ReactDiffNNet(z_dim=z_dim, c_dim=c_dim, device="cuda")
    n_samples = 10
    samples = torch.randn((n_samples, 2 + z_dim, 32, 32)).to("cuda")
    context = torch.randn((n_samples, c_dim)).to("cuda")
    out = nnet(samples, context=context)
    print(out.shape)
