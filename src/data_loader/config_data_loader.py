import os, json
from src.data_loader import get_dataset_loader
from src.data_loader.simulator_dataset import SimulatorDataset
from src.data_loader.reactdiff_dataset import ReactionDiffusionDataset

from src.sampler.distributions import BoxUniformSampler, Bernoulli
from src.sampler.simulator_sampler import SimulationSampler
from src.sampler.advdiff.advdiff_sampler import AdvDiffSampler
from src.sampler.reactdiff.reactdiff_sampler import ReactDiffSampler

from src.forward_models.pendulum import PendulumSolver
from src.forward_models.advdiff import AdvDiffSolver
from src.forward_models.reactdiff.reactdiff_model import ReactDiffSolver

import torch
import numpy as np


def config_data_loader(config, data_cfg, device):
    # Load train data
    tr_path_sims_x = os.path.join(
        data_cfg.datadir, data_cfg.dataname_train_labeled_sims
    )
    if data_cfg["name"] == "reactdiff" and not tr_path_sims_x.endswith(".pt"):
        tr_x_dataset = ReactionDiffusionDataset(
            name_dataset=data_cfg["name"],
            data_file_path=tr_path_sims_x,
            device=device,
        )
        conf_exp_x = tr_x_dataset.conf_data
    else:
        tr_x_dataset = SimulatorDataset(
            name_dataset=data_cfg["name"],
            data_file_path=tr_path_sims_x,
            max_train_size=data_cfg.max_train_size,
            device=device,
            lazy_loading=(
                data_cfg["lazy_loading"]
                if "lazy_loading" in data_cfg
                else False
            ),
        )
        # Configuration for the x data
        tr_path_x_conf = os.path.join(
            os.path.dirname(tr_path_sims_x),
            os.path.basename(tr_path_sims_x).replace("data", "args"),
        )
        tr_path_x_conf_params = os.path.splitext(tr_path_x_conf)[0] + ".json"
        conf_exp_x = json.load(open(tr_path_x_conf_params, "r"))

    tr_path_y = os.path.join(
        data_cfg.datadir, data_cfg.dataname_train_unlabeled_y
    )

    val_path_y = os.path.join(
        data_cfg.datadir,
        data_cfg.dataname_val_unlabeled_y,
    )

    if data_cfg["name"] == "reactdiff" and not tr_path_y.endswith(".pt"):
        tr_y_dataset = ReactionDiffusionDataset(
            name_dataset=data_cfg["name"],
            data_file_path=tr_path_y,
            device=device,
        )
        conf_exp_y = tr_x_dataset.conf_data
    else:
        tr_y_dataset = SimulatorDataset(
            name_dataset=data_cfg["name"],
            data_file_path=tr_path_y,
            device=device,
            lazy_loading=(
                data_cfg["lazy_loading"]
                if "lazy_loading" in data_cfg
                else False
            ),
        )
        # Params configuration for the y data
        tr_path_y_conf_params = os.path.join(
            os.path.dirname(tr_path_y),
            os.path.basename(tr_path_y).replace("data", "args"),
        )
        tr_path_y_conf_params = (
            os.path.splitext(tr_path_y_conf_params)[0] + ".json"
        )
        conf_exp_y = json.load(open(tr_path_y_conf_params, "r"))

    if data_cfg["name"] == "reactdiff" and not val_path_y.endswith(".pt"):
        val_y_dataset = ReactionDiffusionDataset(
            name_dataset=data_cfg["name"],
            data_file_path=val_path_y,
            testing_set=True,
            device=device,
        )
    else:
        val_y_dataset = SimulatorDataset(
            name_dataset=data_cfg["name"],
            data_file_path=val_path_y,
            testing_set=True,
            device=device,
            lazy_loading=(
                data_cfg["lazy_loading"]
                if "lazy_loading" in data_cfg
                else False
            ),
        )

    # Load test data
    te_dataset = None
    tr_batch_size = config.batch_size
    val_batch_size = config.val_batch_size
    if data_cfg["name"] == "reactdiff":
        # Adjust batch size for reactdiff, the dataset already is loaded by batches of 64(default)
        if not tr_path_sims_x.endswith(".pt"):
            tr_batch_size = tr_batch_size // tr_x_dataset.batch_size_data
            config["batch_size"] = tr_batch_size
        if not val_path_y.endswith(".pt"):
            val_batch_size = val_batch_size // val_y_dataset.batch_size_data
            config["val_batch_size"] = val_batch_size

    # Init data loaders
    (
        train_x_loader,
        train_unlabel_loader,
        val_y_loader,
        test_loader,
    ) = get_dataset_loader(
        tr_x_dataset,
        tr_y_dataset,
        val_y_dataset=val_y_dataset,
        batch_size=tr_batch_size,
        val_batch_size=val_batch_size,
        test_dataset=te_dataset,
        device=device,
    )
    return (
        train_x_loader,
        train_unlabel_loader,
        val_y_loader,
        test_loader,
        conf_exp_x,
        conf_exp_y,
    )


def get_x_params_sampler(config, device):
    sampler_config = config["data_sampler"]
    phys_params_cfg_sampler = sampler_config["phys_params_sampler"]
    if phys_params_cfg_sampler["x"]["type"] == "uniform":
        lower_bound = torch.tensor(
            phys_params_cfg_sampler["x"]["lower_bound"],
            device=device,
            dtype=torch.float32,
        )
        upper_bound = torch.tensor(
            phys_params_cfg_sampler["x"]["upper_bound"],
            device=device,
            dtype=torch.float32,
        )
        x_params_sampler = BoxUniformSampler(lower_bound, upper_bound)
    else:
        raise NotImplementedError("Sampler type not supported")
    return x_params_sampler


def get_y_sampler(config, device):
    sampler_config = config["data_sampler"]
    phys_params_cfg_sampler = sampler_config["phys_params_sampler"]
    y_params_sampler = None
    tr_y_sampler, val_y_sampler = None, None
    conf_exp_y = None
    if "y" in phys_params_cfg_sampler:

        lower_bound = torch.tensor(
            phys_params_cfg_sampler["y"]["lower_bound"],
            device=device,
            dtype=torch.float32,
        )
        upper_bound = torch.tensor(
            phys_params_cfg_sampler["y"]["upper_bound"],
            device=device,
            dtype=torch.float32,
        )
        if phys_params_cfg_sampler["y"]["type"] == "uniform":
            y_params_sampler = BoxUniformSampler(lower_bound, upper_bound)
        elif phys_params_cfg_sampler["y"]["type"] == "bernoulli":
            y_params_sampler = Bernoulli(
                head=lower_bound, tail=upper_bound, device=device
            )
    else:
        # Params configuration for the y data
        tr_path_y = os.path.join(
            sampler_config["datadir"],
            sampler_config["dataname_train_unlabeled_y"],
        )
        val_path_y = os.path.join(
            sampler_config["datadir"],
            sampler_config["dataname_val_unlabeled_y"],
        )
        if "dataname_args" in sampler_config.keys():
            tr_path_y_conf_params = os.path.join(
                sampler_config["datadir"],
                sampler_config["dataname_args"],
            )
        else:
            tr_path_y_conf_params = os.path.join(
                os.path.dirname(tr_path_y),
                os.path.basename(tr_path_y).replace("data", "args"),
            )
            tr_path_y_conf_params = (
                os.path.splitext(tr_path_y_conf_params)[0] + ".json"
            )

        conf_exp_y = json.load(open(tr_path_y_conf_params, "r"))

        tr_y_dataset = SimulatorDataset(
            name_dataset=sampler_config["name"],
            data_file_path=tr_path_y,
            device=device,
            lazy_loading=(
                sampler_config["lazy_loading"]
                if "lazy_loading" in sampler_config
                else False
            ),
        )

        val_y_dataset = SimulatorDataset(
            name_dataset=sampler_config["name"],
            data_file_path=val_path_y,
            testing_set=True,
            device=device,
            lazy_loading=(
                sampler_config["lazy_loading"]
                if "lazy_loading" in sampler_config
                else False
            ),
        )

        _, tr_y_sampler, val_y_sampler, _ = get_dataset_loader(
            None,
            tr_y_dataset,
            val_y_dataset=val_y_dataset,
            batch_size=config.batch_size,
            test_dataset=None,
            val_batch_size=config.val_batch_size,
            device=device,
        )

    return y_params_sampler, tr_y_sampler, val_y_sampler, conf_exp_y


def config_phys_sampler(config, device):
    """
    Configures the data sampler for the physical model
    """
    black_box_physics = (
        config["model"]["phys_model"] is None
        or config["model"]["phys_model"] == ""
        or config["model"]["phys_model"].lower() == "none"
        or "black" in config["model"]["phys_model"].lower()
    )

    sampler_config = config["data_sampler"]
    init_cond_cfg_sampler = sampler_config["init_cond_sampler"]
    if init_cond_cfg_sampler["type"] == "uniform":
        l_bound = init_cond_cfg_sampler["lower_bound"]
        u_bound = init_cond_cfg_sampler["upper_bound"]
        if "extend_dim" in init_cond_cfg_sampler:
            l_bound = [l_bound] * np.prod(init_cond_cfg_sampler["extend_dim"])
            u_bound = [u_bound] * np.prod(init_cond_cfg_sampler["extend_dim"])

        lower_bound = torch.tensor(l_bound, device=device, dtype=torch.float32)
        upper_bound = torch.tensor(u_bound, device=device, dtype=torch.float32)
        lower_bound = lower_bound.flatten()
        upper_bound = upper_bound.flatten()
        init_cond_sampler = BoxUniformSampler(
            lower_bound,
            upper_bound,
            device=device,
        )
    else:
        raise NotImplementedError("Sampler type not supported")

    # X params sampler
    phys_params_cfg_sampler = sampler_config["phys_params_sampler"]
    x_params_sampler = get_x_params_sampler(config, device)

    # Y sampler
    y_params_sampler, tr_y_sampler, val_y_sampler, conf_exp_y = get_y_sampler(
        config, device
    )

    cfg_forward_model = sampler_config["forward_model"]
    if cfg_forward_model["name"] == "pendulum":
        x_fwd = PendulumSolver(
            len_episode=cfg_forward_model["f_evals"],
            dt=cfg_forward_model["dt"],
            noise_loc=cfg_forward_model["noise_loc"],
            noise_std=cfg_forward_model["noise_std"],
            method=cfg_forward_model["method"],
        )
        y_fwd = PendulumSolver(
            len_episode=cfg_forward_model["f_evals"],
            dt=cfg_forward_model["dt"],
            noise_loc=cfg_forward_model["noise_loc"],
            noise_std=cfg_forward_model["noise_std"],
            method=cfg_forward_model["method"],
        )
    elif cfg_forward_model["name"] == "advdiff":
        x_fwd = AdvDiffSolver(
            dx=cfg_forward_model["dx"],
            dt=cfg_forward_model["dt"],
            len_episode=cfg_forward_model["f_evals"],
            n_grids=cfg_forward_model["n_grids"],
            method=cfg_forward_model["method"],
            noise_loc=cfg_forward_model["noise_loc"],
            noise_std=cfg_forward_model["noise_std"],
            device=device,
        )
        y_fwd = AdvDiffSolver(
            dx=cfg_forward_model["dx"],
            dt=cfg_forward_model["dt"],
            len_episode=cfg_forward_model["f_evals"],
            n_grids=cfg_forward_model["n_grids"],
            method=cfg_forward_model["method"],
            noise_loc=cfg_forward_model["noise_loc"],
            noise_std=cfg_forward_model["noise_std"],
            device=device,
        )
    elif cfg_forward_model["name"] == "reactdiff":
        x_fwd = ReactDiffSolver(
            t0=cfg_forward_model["t_span"][0],
            t1=cfg_forward_model["t_span"][1],
            dx=cfg_forward_model["mesh_step"],
            len_episode=cfg_forward_model["f_evals"],
            ode_stepsize=(
                cfg_forward_model["ode_stepsize"]
                if "ode_stepsize" in cfg_forward_model
                else None
            ),
            noise_loc=cfg_forward_model["noise_loc"],
            noise_std=cfg_forward_model["noise_std"],
            method=cfg_forward_model["method"],
            device=device,
        )
        y_fwd = ReactDiffSolver(
            t0=cfg_forward_model["t_span"][0],
            t1=cfg_forward_model["t_span"][1],
            dx=cfg_forward_model["mesh_step"],
            len_episode=cfg_forward_model["f_evals"],
            ode_stepsize=(
                cfg_forward_model["ode_stepsize"]
                if "ode_stepsize" in cfg_forward_model
                else None
            ),
            noise_loc=cfg_forward_model["noise_loc"],
            noise_std=cfg_forward_model["noise_std"],
            method=cfg_forward_model["method"],
            device=device,
        )

    # training sampler
    if black_box_physics:
        x_fwd = None
    if cfg_forward_model["name"] == "advdiff":
        n_grids = cfg_forward_model["n_grids"]
        dx = cfg_forward_model["dx"]
        tr_x_sampler = AdvDiffSampler(
            init_cond_sampler=init_cond_sampler,
            params_sampler=x_params_sampler,
            sim_solver=x_fwd,
            n_grids=n_grids,
            dx=dx,
            batch_size=config["batch_size"],
            device=device,
        )
    elif cfg_forward_model["name"] == "reactdiff":
        tr_x_sampler = ReactDiffSampler(
            init_cond_sampler=init_cond_sampler,
            params_sampler=x_params_sampler,
            sim_solver=x_fwd,
            batch_size=config["batch_size"],
            device=device,
        )
    else:
        tr_x_sampler = SimulationSampler(
            init_cond_sampler=init_cond_sampler,
            params_sampler=x_params_sampler,
            sim_solver=x_fwd,
            batch_size=config["batch_size"],
            device=device,
        )  #

    if tr_y_sampler is None:
        if cfg_forward_model["name"] == "advdiff":
            n_grids = cfg_forward_model["n_grids"]
            dx = cfg_forward_model["dx"]
            tr_y_sampler = AdvDiffSampler(
                init_cond_sampler=init_cond_sampler,
                params_sampler=y_params_sampler,
                sim_solver=y_fwd,
                n_grids=n_grids,
                dx=dx,
                batch_size=config["batch_size"],
                device=device,
            )
        elif cfg_forward_model["name"] == "reactdiff":
            tr_y_sampler = ReactDiffSampler(
                init_cond_sampler=init_cond_sampler,
                params_sampler=y_params_sampler,
                sim_solver=y_fwd,
                batch_size=config["batch_size"],
                device=device,
            )
        else:
            z_size = config["data_sampler"]["phys_params_sampler"]["y"][
                "z_size"
            ]
            tr_y_sampler = SimulationSampler(
                init_cond_sampler=init_cond_sampler,
                params_sampler=y_params_sampler,
                x_params_sampler=x_params_sampler,
                params_z_size=z_size,
                sim_solver=y_fwd,
                batch_size=config["batch_size"],
                device=device,
            )

    # validation sampler
    if val_y_sampler is None:
        if cfg_forward_model["name"] == "advdiff":
            n_grids = cfg_forward_model["n_grids"]
            dx = cfg_forward_model["dx"]
            val_y_sampler = AdvDiffSampler(
                init_cond_sampler=init_cond_sampler,
                params_sampler=y_params_sampler,
                sim_solver=y_fwd,
                n_grids=n_grids,
                dx=dx,
                batch_size=config["val_batch_size"],
                n_val_batches=phys_params_cfg_sampler["n_val_batches"],
                device=device,
            )
        elif cfg_forward_model["name"] == "reactdiff":
            val_y_sampler = ReactDiffSampler(
                init_cond_sampler=init_cond_sampler,
                params_sampler=y_params_sampler,
                sim_solver=y_fwd,
                batch_size=config["val_batch_size"],
                n_val_batches=phys_params_cfg_sampler["n_val_batches"],
                device=device,
            )
        else:
            z_size = config["data_sampler"]["phys_params_sampler"]["y"][
                "z_size"
            ]
            val_y_sampler = SimulationSampler(
                init_cond_sampler=init_cond_sampler,
                params_sampler=y_params_sampler,
                x_params_sampler=x_params_sampler,
                params_z_size=z_size,
                sim_solver=y_fwd,
                testing_set=True,
                batch_size=config["val_batch_size"],
                n_val_batches=phys_params_cfg_sampler["n_val_batches"],
                device=device,
            )

    te_y_sampler = None

    conf_exp_x = {
        "init_cond_sampler": sampler_config["init_cond_sampler"],
        "phys_params_sampler": sampler_config["phys_params_sampler"]["x"],
        "forward_model": sampler_config["forward_model"],
    }
    if conf_exp_y is None:
        conf_exp_y = {
            "init_cond_sampler": sampler_config["init_cond_sampler"],
            "phys_params_sampler": {
                sampler_config["phys_params_sampler"]["y"]
            },
            "forward_model": sampler_config["forward_model"],
        }

    return (
        tr_x_sampler,
        tr_y_sampler,
        val_y_sampler,
        te_y_sampler,
        conf_exp_x,
        conf_exp_y,
    )


def get_sims_loader(config, device):
    if "data_sampler" in config and config["data_sampler"]["online"]:
        return config_phys_sampler(config, device)
    else:
        data_cfg = (
            config["data_sampler"] if "data_sampler" in config else config
        )
        return config_data_loader(config, data_cfg, device)
