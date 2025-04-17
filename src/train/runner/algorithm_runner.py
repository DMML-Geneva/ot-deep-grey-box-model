import gc
import os
import time

import logging
import yaml
from omegaconf import OmegaConf, DictConfig

import numpy as np

import torch.utils.data
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import get_uuid_hash_from_task_exp
from src.train.ot_physics.train.knot_train import knot_training
from src.train.gans import train_gans, train_wgans
from src.utils import hash
from src.nnets import wandb_util
from src.nnets.utils import load_state_model
from src.train.ot_physics.init_nnet_task import init_nnets_task

import wandb

from src.data_loader.config_data_loader import get_sims_loader


def train(
    config,
    T,
    f,
    T_opt,
    f_opt,
    T_scheduler,
    f_scheduler,
    train_x_loader,
    train_unlabel_loader,
    val_unlabel_loader,
    logger,
    out_path_chkp,
    wandb,
):
    # Train the model

    if "algorithm_name" in config and "wgans" in config["algorithm_name"]:
        res = train_wgans(
            train_x_sampler=train_x_loader,
            train_y_sampler=train_unlabel_loader,
            val_y_sampler=val_unlabel_loader,
            gen_T=T,
            disc_f=f,
            args=config,
            wandb=wandb,
            logger=logger,
            out_path_chkp=out_path_chkp,
        )
    elif "algorithm_name" in config and "gans" == config["algorithm_name"]:
        res = train_gans(
            train_x_sampler=train_x_loader,
            train_y_sampler=train_unlabel_loader,
            val_y_sampler=val_unlabel_loader,
            gen_T=T,
            disc_f=f,
            args=config,
            wandb=wandb,
            logger=logger,
            out_path_chkp=out_path_chkp,
        )
    else:
        res = knot_training(
            train_x_loader,
            train_unlabel_loader,
            val_unlabel_loader,
            T,
            f,
            T_opt=T_opt,
            f_opt=f_opt,
            T_scheduler=T_scheduler,
            f_scheduler=f_scheduler,
            args=config,
            wandb=wandb,
            logger=logger,
            out_path_chkp=out_path_chkp,
        )

    return res


def run_algorithm(config):
    SEED = config.seed

    # Set up logging
    logging.basicConfig(level=config.logging)
    logger = logging.getLogger("ADV-KNOT")
    logger.setLevel(config.logging)

    # Set the CUDA device
    device = config.device if "device" in config else None
    if (device is None and torch.cuda.is_available()) or device == "cuda":
        num_gpus = torch.cuda.device_count()
        DEVICE_IDS = list(range(num_gpus))
        # set and get device
        torch.cuda.set_device(f"cuda:{DEVICE_IDS[0]}")
        device = torch.device(f"cuda:{DEVICE_IDS[0]}")
        logger.info(
            f"Using GPU: {torch.cuda.get_device_name(device)}, idx: {DEVICE_IDS[0]}"
        )
    else:
        logger.warn("CUDA is not available. Using CPU")
        device = torch.device("cpu")
        DEVICE_IDS = []

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.cuda.empty_cache()
    gc.collect()

    # check if training requires a bb model or a grey-box model
    black_box_physics = (
        config["model"]["phys_model"] is None
        or config["model"]["phys_model"] == ""
        or config["model"]["phys_model"].lower() == "none"
        or "black" in config["model"]["phys_model"].lower()
    )

    # Load data
    logger.info("Loading data")

    (
        train_x_loader,
        train_unlabel_loader,
        val_unlabel_loader,
        test_loader,
        conf_exp_x,
        conf_exp_y,
    ) = get_sims_loader(config, device)

    # Random sample from training set
    sample_dict = train_x_loader.sample(1)
    rnd_x_sample = sample_dict["x"]

    if black_box_physics:
        # only the initial condition is used
        obs_dim = sample_dict["init_conds"].shape[1:]
        sample_y = train_unlabel_loader.sample(1)["x"]
        out_dim = sample_y.shape[1:]
        train_unlabel_loader.reset()
    else:
        obs_dim = rnd_x_sample.shape[1:]
        out_dim = obs_dim
    params_dim = sample_dict["params"].shape[1:]
    # Reset the data loaders
    train_x_loader.reset()

    # Initialize the model
    T, f = init_nnets_task(
        config,
        obs_dim,
        params_dim,
        out_dim,
        device,
        logger,
    )

    # Initialize the optimizers
    learning_rates = config.lr
    T_scheduler = None
    f_scheduler = None
    if "optimizer" in config and config["optimizer"] == "sgd":
        T_opt = torch.optim.SGD(
            T.parameters(),
            lr=learning_rates[0],
            weight_decay=1e-2,
        )
        f_opt = torch.optim.SGD(
            f.parameters(), lr=learning_rates[1], weight_decay=1e-3
        )
        T_scheduler = torch.optim.lr_scheduler.LinearLR(
            T_opt,
            start_factor=1.0,
            end_factor=1e-5,
            total_iters=config["max_epochs"],
        )
        f_scheduler = torch.optim.lr_scheduler.LinearLR(
            T_opt,
            start_factor=1.0,
            end_factor=1e-5,
            total_iters=config["max_epochs"],
        )
    else:
        T_opt = torch.optim.Adam(
            T.parameters(),
            lr=learning_rates[0],
            betas=tuple(config.betas_lr),
            weight_decay=1e-10,
        )

        f_opt = torch.optim.Adam(
            f.parameters(),
            lr=learning_rates[1],
            betas=tuple(config.betas_lr),
            weight_decay=1e-10,
        )

    # Check if the model is a pre-trained and load it.
    if "pretrained_model" in config and config.pretrained_model is not None:
        check_pt = config.pretrained_model
        if os.path.exists(check_pt):
            T, f, T_opt, f_opt, dic_chkpt = load_state_model(
                T, f, T_opt, f_opt, check_pt
            )
            logger.info(f"Pre-trained model loaded: {check_pt}")
        else:
            raise ValueError(f"Pre-trained model not found: {check_pt}")

    if "lr_scheduler" in config and config["lr_scheduler"] is not None:
        T_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            T_opt,
            milestones=config["lr_scheduler"]["milestones"],
            gamma=config["lr_scheduler"]["gamma"],
        )
        f_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            T_opt,
            milestones=config["lr_scheduler"]["milestones"],
            gamma=config["lr_scheduler"]["gamma"],
        )

    # Hash the experiment datasets
    # The experiment is unique by the x_params_conf and y_params_conf
    u_x, u_y = get_uuid_hash_from_task_exp(
        conf_exp_x=conf_exp_x, conf_exp_y=conf_exp_y, config=config
    )

    group_exp_name = f"phys-ot_{u_x}_{u_y}"

    logger.info(f"Experiment datasets hashes: x = {u_x}, y = {u_y}")
    logger.info(f"Group experiment name: {group_exp_name}")

    # Create hash of the model options
    model_hash, _ = hash.get_uuid_hash_from_config(
        config, config.MODEL_KEYS
    )  #
    salt = hash.generate_salt()
    model_exp_name = f"model_{model_hash}_exp_{u_x}_{u_y}_salt_{salt}"
    logger.info(f"Model experiment name: {model_exp_name}")

    out_path_chkp = "{}/checkpoints/{}/{}".format(
        config.outdir, salt, model_exp_name
    )
    dir_chkp = os.path.dirname(out_path_chkp)
    if not os.path.exists(dir_chkp):
        os.makedirs(dir_chkp)

    # save config file
    config_name = f"{model_exp_name}_config.yaml"
    with open(f"{dir_chkp}/{config_name}", "w") as dump_file:
        if isinstance(config, DictConfig):
            cf_to_save = OmegaConf.to_container(config, resolve=True)
        elif not isinstance(config, dict):
            cf_to_save = vars(config)
        else:
            cf_to_save = config
        yaml.dump(cf_to_save, dump_file)

    # Setup wandb
    os.environ["WANDB__SERVICE_WAIT"] = "1200"
    logger.info("Initializing wandb...")
    # To be safe, finish any previous run:
    if (
        config.wandb_api_key is None
        or config.wandb_username is None
        or config.wandb_username == ""
    ):
        wandb.finish()
    else:
        while True:
            try:
                wandb.finish()
                wandb.login(key=config.wandb_api_key, relogin=True)
                tags = [
                    config["algorithm_name"],
                    (
                        config["model"]["model_name"]
                        if "model_name" in config
                        else config["model"]["type"]
                    ),
                    config["model"]["nnet_arch"],
                ]

                tags.append("black-box" if black_box_physics else "grey-box")
                wandb.init(
                    name=model_exp_name,
                    project="phys-adv-knot",
                    entity=config.wandb_username,
                    group=f"{group_exp_name}",
                    tags=tags,
                    config=(
                        vars(config)
                        if not isinstance(config, DictConfig)
                        else OmegaConf.to_container(config, resolve=True)
                    ),
                )

            except (wandb.errors.CommError, ConnectionRefusedError) as e:
                logger.error(f"Error in wandb: {e}")
                time.sleep(5)
                continue

            logger.info("Wandb initialized")
            logger.info(f"Model experiment name: {model_exp_name}")
            logger.info(f"Group experiment name: {group_exp_name}")
            logger.info(f"Configuration: {config}")
            break

    if wandb.run is not None and config.wandb_model_watch:
        wandb.log({"seed": SEED})
        wandb_util.watch(
            {
                "T generator": T,
                "f discriminator": f,
            },
            log="all",
            log_freq=config.t_iters,
        )
        logger.info("Wandb watch initialized for T and f")

    # Train the model
    res = train(
        config,
        T,
        f,
        T_opt,
        f_opt,
        T_scheduler,
        f_scheduler,
        train_x_loader,
        train_unlabel_loader,
        val_unlabel_loader,
        logger,
        out_path_chkp,
        wandb,
    )
    T, f = res["T"], res["f"]

    # Evaluate the mode

    # evaluate(
    #    config, config.outdir, model_exp_name, physics_model, sample_size=-1
    # )

    if wandb.run is not None:
        wandb.finish()

    return res
