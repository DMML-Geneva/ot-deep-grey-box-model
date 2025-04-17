import torch
from torch import nn
import numpy as np

from src.forward_models.pendulum import SimplePendulum

from src.nnets.mlp import MLPConditionalGenerator
from src.nnets.neuralode.pendulum_adjodenet import (
    NeuralPendulum as NeuralPendulumAdj,
)
from src.nnets.neuralode.pendulum_odenet import NeuralPendulum
from src.nnets.neuralode.advdiff_odenet import NeuralAdvDiff, AdvDiffNNet
from src.nnets.neuralode.reactdiff_odenet import NeuralReactDiff


def odenet_builder(
    task_name,
    z_dim,
    params_dim,
    config=None,
    device="cuda",
):

    dt = config["data_sampler"]["forward_model"]["dt"]
    len_episode = config["data_sampler"]["forward_model"]["f_evals"]

    if task_name == "pendulum":
        t_intg = torch.linspace(0.0, dt * (len_episode - 1), len_episode).to(
            device
        )
        state_dim = 1
        input_dim = (
            state_dim + 1
        )  #  x(t) (current state), \alpha(t), optionally (t)
        if (
            "time_dim_flag" not in config["model"]
            or config["model"]["time_dim_flag"]
        ):
            input_dim += 1

        nnet = MLPConditionalGenerator(
            input_dim,
            z_dim=z_dim,
            activation=(
                config["model"]["activation"]
                if "activation" in config["model"]
                else "relu"
            ),
            c_dim=params_dim,  # params
            z_std=0.1,
            output_dim=state_dim,
            h_layers=config["model"]["h_layers"],
            device=device,
        )

        # Divide each initialized weight by the number of function evaluation of the neural ode!
        for p in nnet.parameters():
            factor = len_episode  #
            p.data = p.data / factor

        physics_model = SimplePendulum().to(device)
        if config["model"]["adjoint_solver"]:
            odenet = NeuralPendulumAdj(
                nnet,
                physics_model,
                t_intg=t_intg,
                cfg_model=config["model"],
                device=device,
            )
        else:
            odenet = NeuralPendulum(
                nnet,
                physics_model,
                ode_solver=(
                    config["model"]["ode_solver"]
                    if "ode_solver" in config["model"]
                    else "euler"
                ),
                t_intg=t_intg,
                cfg_model=config["model"],
                device=device,
            )

    elif task_name == "advdiff":
        dx = config["data_sampler"]["forward_model"]["dx"]
        n_grids = config["data_sampler"]["forward_model"]["n_grids"]

        state_dim = n_grids
        c_dim = 1 + 1 + 1  # params_dim=1 + sin(t), cos(t)
        # Initialize the models
        cfg_model = config["model"]
        nnet = AdvDiffNNet(
            state_dim=state_dim,
            z_dim=z_dim,
            c_dim=c_dim,  # params
            conv_nfilters=(
                cfg_model["conv_nfilters"]
                if "conv_nfilters" in cfg_model
                else 4
            ),
            activation=(
                cfg_model["activation"]
                if "activation" in config["model"]
                else "relu"
            ),
            output_dim=n_grids,  # output_dim
            h_layers=config["model"]["h_layers"],
            device=device,
        )

        odenet = NeuralAdvDiff(
            dx=dx,
            dt=dt,
            len_episode=len_episode,
            n_grids=n_grids,
            nnet=nnet,
            cfg_model=config["model"],
            store_time_eval=False,
            device=device,
        )
    elif task_name == "reactdiff":
        n_grids = config["data_sampler"]["forward_model"]["grid_size"]
        len_episode = config["data_sampler"]["forward_model"]["f_evals"]
        state_dim = n_grids
        c_dim = params_dim  # params_dim + sin(t), cos(t)
        odenet = NeuralReactDiff(
            t0=config["data_sampler"]["forward_model"]["t_span"][0],
            t1=config["data_sampler"]["forward_model"]["t_span"][1],
            dx=config["data_sampler"]["forward_model"]["mesh_step"],
            len_episode=len_episode,
            z_dim=z_dim,
            cfg_model=config["model"],
            store_time_eval=False,
            device=device,
        )
        nnet = odenet.nnet
        # Divide each initialized weight by the number of function evaluation of the neural ode!
        for p in nnet.parameters():
            factor = len_episode * 4
            p.data = p.data / factor

    return {"nnet": nnet, "odenet": odenet}
