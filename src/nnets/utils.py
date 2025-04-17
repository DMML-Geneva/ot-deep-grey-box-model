import os

import yaml

import torch
from torch import nn

import math

from src.nnets.mlp import MLPConditionalGenerator
from src.nnets.neuralode.odenet_builder import odenet_builder
from src.train.ot_physics.init_nnet_task import init_nnets_task


def save_state_model(T, f, epochs, T_optimizer, f_optimizer, out_path):
    torch.save(
        {
            "epoch": epochs,
            "T_model_state_dict": T.state_dict(),
            "f_model_state_dict": f.state_dict(),
            "T_optimizer_state_dict": T_optimizer.state_dict(),
            "f_optimizer_state_dict": f_optimizer.state_dict(),
        },
        out_path,
    )


def load_state_model(T, f, T_optimizer, f_optimizer, in_path):
    if os.path.exists(in_path):
        checkpoint = torch.load(in_path)

    if T is not None:
        T.load_state_dict(checkpoint["T_model_state_dict"])

    if f is not None:
        f.load_state_dict(checkpoint["f_model_state_dict"])

    if T_optimizer is not None:
        T_optimizer.load_state_dict(checkpoint["T_optimizer_state_dict"])
    if f_optimizer is not None:
        f_optimizer.load_state_dict(checkpoint["f_optimizer_state_dict"])

    return T, f, T_optimizer, f_optimizer, checkpoint


def load_model(
    task_name=None,
    dir_model=None,
    model_name=None,
    path_model_file=None,
    path_config_file=None,
    load_best_model=True,
    params_dim=1,
    x_dim=None,
    init_cond_dim=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    if path_model_file is None:
        parent_dir = f"{dir_model}/checkpoints/{model_name[-4:]}"
        # Initialize the models
        model_conf_path = f"{parent_dir}/{model_name}_config.yaml"
    else:
        parent_dir = os.path.dirname(path_model_file)
        if path_config_file is not None:
            model_conf_path = path_config_file
        else:
            model_conf_path = path_model_file.replace(
                "_best.pt", "_config.yaml"
            )
            model_conf_path = model_conf_path.replace(
                "_final.pt", "_config.yaml"
            )
    print(f"Loading model from {model_conf_path}")
    with open(model_conf_path, "r") as file_config:
        config = yaml.load(file_config, Loader=yaml.FullLoader)

    # For predator_prey, it should be 4* 3=12 because of the time embedding! BUt it's having 9 for some reason. Might be time embedding configuration not saved correctly. check it in debug mode!
    det_model = config["model"]["type"] == "det"
    if config["model"]["nnet_arch"] == "odenet":
        if det_model > 0:
            z_dim = 0
        else:
            z_dim = config["z_dim"] if "z_dim" in config else 1

        res_builder = odenet_builder(
            task_name=config["data_sampler"]["name"],
            z_dim=z_dim,
            params_dim=params_dim,
            config=config,
            device=device,
        )
        nnets = res_builder["nnet"]
        T_model = res_builder["odenet"]
    else:
        length_episodes = config["data_sampler"]["forward_model"]["f_evals"]
        obs_dim = length_episodes
        if det_model > 0:
            z_dim = 0
        else:
            z_dim = config["z_dim"] if "z_dim" in config else obs_dim

        black_box = (
            config["model"]["phys_model"] == "none"
            or config["model"]["phys_model"] == "black_box"
        )
        if black_box:
            output_dim = obs_dim
            x_dim = init_cond_dim
        else:
            output_dim = obs_dim

        if task_name == "advdiff":
            params_dim = params_dim * init_cond_dim
            output_dim = (init_cond_dim, output_dim)
            if not black_box:
                x_dim = init_cond_dim * obs_dim

        T_model, disc_net = init_nnets_task(
            config=config,
            input_dshape=(x_dim,),
            params_dim=params_dim,
            out_dim_shape=(
                output_dim if isinstance(output_dim, tuple) else (output_dim,)
            ),
            device=device,
        )

    print(T_model)

    # Load the saved model
    if path_model_file is None:
        if load_best_model:
            checkpoint = f"{parent_dir}/T_{model_name}_best.pt"
            if os.path.exists(checkpoint):
                model_dict = torch.load(checkpoint, map_location=device)
                T_model.load_state_dict(model_dict)
                dic_chkpt = model_dict
            else:
                checkpoint = f"{parent_dir}/{model_name}_best.pt"
                if os.path.exists(checkpoint):
                    (T_model, _, _, _, dic_chkpt) = load_state_model(
                        T_model, None, None, None, checkpoint
                    )
                else:
                    print("Model not found")
                    return None, None, None, None, None
        else:
            if os.path.exists(path_model_file):
                model_dict = torch.load(path_model_file, map_location=device)
                T_model.load_state_dict(model_dict)
                dic_chkpt = model_dict
            else:
                checkpoint = f"{parent_dir}/{model_name}_final.pt"
                if os.path.exists(checkpoint):
                    (T_model, _, _, _, dic_chkpt) = load_state_model(
                        T_model, None, None, None, checkpoint
                    )
                else:
                    print("Model not found")
                    return None, None, None, None, None
    else:
        print(f"Loading model from {path_model_file}")
        model_dict = torch.load(path_model_file, map_location=device)
        dic_chkpt = model_dict
        try:
            if "T_model_state_dict" in model_dict:
                T_model.load_state_dict(model_dict["T_model_state_dict"])
            else:
                T_model.load_state_dict(model_dict)
        except Exception as e:
            print(f"Error loading state model: {e}")
            return None, None, None, None

    return T_model, parent_dir, config, dic_chkpt


def get_conv1d_output_dim(
    input_dim, kernel_size, stride=1, padding=0, dilation=1
):
    return int(
        ((input_dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride)
        + 1
    )
