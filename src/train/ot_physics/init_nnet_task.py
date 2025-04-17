from src.nnets.neuralode.odenet_builder import odenet_builder
import torch
import torch.nn as nn
import math

from src.nnets import LSTMNet
from src.nnets import MLPConditionalGenerator
from src.nnets import get_mlp_discriminator
from src.nnets import Conv1DResidualNet
from src.nnets.unet_physics import (
    UNetReactionAttentionDiffusion,
    UNetReactionDiffusion,
    UNetReaction3DDiffusion,
)


import numpy as np
import torch


def weights_init_nnet(m, activation="leaky_relu"):

    if (
        isinstance(m, nn.Conv1d)
        or isinstance(m, nn.Conv2d)
        or isinstance(m, nn.Linear)
    ):
        if activation == "relu":
            nn.init.kaiming_normal_(
                m.weight, mode="fan_in", nonlinearity="relu"
            )
        elif activation == "leaky_relu":
            nn.init.kaiming_normal_(
                m.weight, mode="fan_in", nonlinearity="leaky_relu"
            )
        elif activation == "elu":
            nn.init.kaiming_normal_(
                m.weight,
                mode="fan_in",
                nonlinearity="leaky_relu",  # same as leaky_relu
            )
        elif activation == "selu":
            nn.init.kaiming_normal_(
                m.weight, mode="fan_in", nonlinearity="selu"
            )
        elif activation == "tanh":
            nn.init.xavier_normal_(m.weight)
        elif activation == "softplus":
            nn.init.xavier_normal_(m.weight)
        elif activation == "sine":

            def init_sine_weights(m):
                std = 1.0 / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-2.0 * std, 2.0 * std)
                m.bias.data.zero_()

            init_sine_weights(
                m
            )  # As the implementation of Neural ODEs in torchdiffeq in learn_physics.py

    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def init_nnets_task(
    config,
    input_dshape,
    params_dim,
    out_dim_shape,
    device,
    logger=None,
):
    input_dim = np.prod(input_dshape)
    params_dim = np.prod(params_dim)

    if out_dim_shape is None or out_dim_shape == 0:
        output_dim = input_dim
    else:
        output_dim = np.prod(out_dim_shape)

    det_model = (
        "model_name" in config["model"]
        and config["model"]["model_name"] == "not"
    ) or ("type" in config["model"] and config["model"]["type"] == "det")

    black_box_physics = (
        config["model"]["phys_model"] is None
        or config["model"]["phys_model"] == ""
        or config["model"]["phys_model"].lower() == "none"
        or "black" in config["model"]["phys_model"].lower()
    )

    # Generator architecture
    if config["model"]["nnet_arch"] == "mlp":
        if det_model > 0:
            z_dim = 0
        else:
            z_dim = config["z_dim"] if "z_dim" in config else input_dim

        T = MLPConditionalGenerator(
            x_dim=input_dim,
            c_dim=params_dim,
            output_dim=output_dim,
            h_layers=config["model"]["h_layers"],
            z_dim=z_dim,
            z_std=0.1,
            activation=(
                config["model"]["activation"]
                if "activation" in config["model"]
                else "relu"
            ),
            with_drop_out=(
                config["model"]["with_drop_out"]
                if "with_drop_out" in config["model"]
                else False
            ),
            with_layer_norm=(
                config["model"]["with_layer_norm"]
                if "with_layer_norm" in config["model"]
                else False
            ),
            device=device,
        )
        for p in T.parameters():
            factor = input_dim  #
            p.data = p.data / factor

    elif config["model"]["nnet_arch"] == "conv1dresnet":
        if det_model > 0:
            z_dim = 0
        else:
            z_dim = config["z_dim"] if "z_dim" in config else input_dim
        T = Conv1DResidualNet(
            x_dim=input_dim + z_dim,
            z_dim=z_dim,
            out_dim=input_dim,
            context_dim=params_dim,
            nlayers=config["model"]["h_layers"],
            device=device,
        ).to(device)
        activation_str = T.activation_str
        init_weight = lambda m: weights_init_nnet(m, activation=activation_str)
        T.apply(init_weight)

    elif config["model"]["nnet_arch"] == "odenet":
        if det_model > 0:
            z_dim = 0
        else:
            z_dim = config["z_dim"] if "z_dim" in config else 1
        print(params_dim)
        res_builder = odenet_builder(
            task_name=config["data_sampler"]["name"],
            z_dim=z_dim,
            params_dim=params_dim,
            config=config,
            device=device,
        )
        T = res_builder["odenet"]
    elif config["model"]["nnet_arch"] == "lstm":
        if det_model > 0:
            z_dim = 0
        else:
            z_dim = config["z_dim"] if "z_dim" in config else input_dim

        if black_box_physics:
            x_shape = input_dshape
        else:
            x_shape = (
                input_dshape[:-1] if len(input_dshape) > 1 else (1,)
            )  # 1 dim, e.g. 1D data (pendulum)

        T = LSTMNet(
            x_shape=x_shape,
            length_seq=out_dim_shape[-1],
            out_shape=out_dim_shape[:-1] if len(out_dim_shape) > 1 else (1,),
            c_dim=params_dim,
            h_dim=config["model"]["h_layers"][0],
            z_dim=z_dim,
            c_embed_dim=(
                config["model"]["c_embed_dim"]
                if "c_embed_dim" in config["model"]
                else params_dim
            ),
            device=device,
        )
    elif config["data_sampler"]["name"] == "reactdiff":
        if config["model"]["nnet_arch"] == "unet":
            T = UNetReactionDiffusion(
                t_dim=16,
                cond_dim=params_dim,
                z_dim=0 if det_model > 0 else config["z_dim"],
                device=device,
            )
        elif config["model"]["nnet_arch"] == "unet3d":
            T = UNetReaction3DDiffusion(
                cond_dim=params_dim,
                z_dim=0 if det_model > 0 else config["z_dim"],
                device=device,
            )
    else:
        raise ValueError("nnet_arch not recognized")

    # Discriminator architecture
    sigmoid = False
    if "algorithm_name" in config and config["algorithm_name"] == "gans":
        sigmoid = True

    if config["data_sampler"]["name"] == "advdiff" or (
        "discriminator" in config["model"]
        and config["model"]["discriminator"]["type"] == "conv1dresnet"
    ):  # or  config["data_sampler"]["name"] == "pendulum"
        f = Conv1DResidualNet(
            x_dim=output_dim,
            nlayers=4,
            sigmoid=sigmoid,
            device=device,
        ).to(device)
        activation_str = f.activation_str
        init_weight = lambda m: weights_init_nnet(m, activation=activation_str)
        f.apply(init_weight)
    elif (
        "discriminator" in config["model"]
        and "unet" in config["model"]["discriminator"]["type"]
    ):
        if config["data_sampler"]["name"] == "reactdiff":
            if config["model"]["discriminator"]["type"] == "unet":
                f = UNetReactionDiffusion(t_dim=16, cond_dim=0, device=device)
            elif config["model"]["discriminator"]["type"] == "unet3d":
                f = UNetReaction3DDiffusion(cond_dim=0, device=device)
        else:
            pass
        activation_str = f.activation_str
        init_weight = lambda m: weights_init_nnet(m, activation=activation_str)
        f.apply(init_weight)
    else:
        input_f_dim = output_dim
        if config["cost"] == "energy-ang-to-euc":
            input_f_dim = output_dim * 2
        if "discriminator" in config["model"]:
            f = get_mlp_discriminator(
                input_f_dim,
                h_layers=config["model"]["discriminator"]["h_layers"],
                device=device,
                sigmoid=sigmoid,
            )
    # convert the above two prints in logging
    if logger is not None:
        logger.info(
            "T params: {}".format(
                np.sum([np.prod(p.shape) for p in T.parameters()])
            )
        )
        logger.info(
            "f params: {}".format(
                np.sum([np.prod(p.shape) for p in f.parameters()])
            )
        )
    return T, f
