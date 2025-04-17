import torch
from torch import nn


EPOCHS = 500
N_CRITIC = 1
CLIP = 0.01

import numpy as np

import torch.utils.data
import torch

import gc
import os

from src.train.ot_physics.utils import (
    unfreeze,
    freeze,
    wandb_log,
    save_checkpoints,
)

from src.evaluation.evaluator import (
    eval_numerical_cond_samples,
)

from .utils import dkl, shannon_entropy

from src.sampler.distributions import sample_latent_noise
from src.forward_models.pendulum.visualization import plot_traj
from src.metrics.mmd import MMDLoss, RBF, estimate_mmd_bandwidth
from tqdm import tqdm_notebook as tqdm
import time


MODEL_KEYS = [
    "model_name",
    "phys_model",
    "cost",
    "nnet_arch",
    "activation",
    "type",
    "n_hidden_layers",
    "h_layers",
    "z_size",
    "z_dim",
    "z_dist",
    "z_std",
    "grad_regularizer",
    "grad_penalty",
    "t_iters",
    "disc_iters",
    "lr",
    "max_epochs",
    "batch_size",
    "alpha",
    "gamma",
    "gamma_iters",
    "discriminator",
]


def sample_from_generator(gen_T, X, X_params, z_size=6, args=None):
    x_dim = X.shape[1:]

    if gen_T.z_dim > 0:
        with torch.no_grad():
            Z = sample_latent_noise(
                n_samples=X.shape[0],
                z_dim=gen_T.z_dim,
                z_size=z_size,
                type=args["z_dist"] if "z_dist" in args else "gauss",
                device=X.device,
            )
            X = X.reshape((-1, 1) + x_dim).repeat(
                [1, z_size] + [1] * len(x_dim)
            )
            X_params = X_params.reshape((-1, 1) + X_params.shape[1:]).repeat(
                [1, z_size] + [1] * len(X_params.shape[1:])
            )
            X_params = X_params.flatten(start_dim=0, end_dim=1)

            if X.ndim > 3:
                Z = Z.reshape(-1, z_size, 1, gen_T.z_dim).repeat(
                    (1, 1) + x_dim[:-1] + (1,)
                )
            X_input = torch.cat([X, Z], dim=-1).flatten(start_dim=0, end_dim=1)

    else:
        X_input = X

    if args["model"]["nnet_arch"] == "odenet":
        preds = gen_T.simulate(X_input, context=X_params)
    else:
        preds = gen_T(X_input, context=X_params)

    return preds


def track_f_stats(
    f_stats_dict,
    f_X=None,
    f_Y=None,
    loss_value=None,
    wandb=None,
    step=None,
    logger=None,
    compute_mean=False,
    validation_stats=False,
):
    with torch.no_grad():
        if f_stats_dict is None:
            f_stats_dict = {
                "tr_f_loss": [],
                "tr_X_entropy": [],
                "tr_Y_entropy": [],
                # "tr_dkl":[],
                # "tr_log_lik": [],
            }

        if not compute_mean:
            f_stats_dict["tr_X_entropy"].append(
                shannon_entropy(f_X.detach()).cpu().tolist()
            )
            f_stats_dict["tr_Y_entropy"].append(
                shannon_entropy(f_Y.detach()).cpu().tolist()
            )
            # f_stats_dict["tr_dkl"].append(dkl(f_X, f_Y))
        else:
            for key in f_stats_dict.keys():
                f_stats_dict[key] = np.mean(f_stats_dict[key])

    if loss_value is not None:
        f_stats_dict["tr_f_loss"].append(loss_value)

    if wandb is not None:
        if validation_stats:
            # copy and replace keys tr_ with val_
            val_f_stats_dict = {}
            for key in f_stats_dict.keys():
                val = f_stats_dict[key]
                if isinstance(val, list) and len(val) > 0:
                    val = np.mean(val)
                val_f_stats_dict[key.replace("tr_", "val_")] = val

            f_stats_dict = val_f_stats_dict
        # send on wandb
        wandb_log(wandb, f_stats_dict, step=step)
        if logger is not None:
            logger.info(f"Step: {step}, f_stats: {f_stats_dict}")

    return f_stats_dict


def train_gans(
    train_x_sampler,
    train_y_sampler,
    val_y_sampler,
    gen_T,
    disc_f,
    args,
    wandb,
    logger,
    out_path_chkp=None,
):
    # Generator Noise parameters
    # Generator Noise parameters
    stoch_map = True if gen_T.z_dim > 0 else False

    Z_SIZE = args.z_size

    num_gpus = torch.cuda.device_count()
    logger.info(
        f"Starting training GAN for {'stochastic' if stoch_map else 'detterministic'} tasks"
    )
    logger.info(
        f"Selected device: {train_x_sampler.device} among {num_gpus} GPUs"
    )
    logger.info(f"Generator model: \n\t{gen_T}")
    logger.info(f"Discriminator model: \n\t{disc_f}")

    logger.info(f"Training for {args.max_epochs} epochs")
    logger.info(f"BATCH_SIZE: {args.batch_size}")

    total_time = time.time()
    best_loss = torch.inf
    simulations_budget = 0

    # estimmate MMD bandwidth
    bandwidth = estimate_mmd_bandwidth(val_y_sampler, median_heuristic=True)
    rbf = RBF(bandwidth=bandwidth, n_kernels=6, device=val_y_sampler.device)
    metrics = [MMDLoss(kernel=rbf)]

    # Init Optimizers
    gen_opt = torch.optim.Adam(gen_T.parameters(), lr=0.0001, betas=(0.0, 0.9))
    disc_f_opt = torch.optim.Adam(
        disc_f.parameters(), lr=0.0001, betas=(0.0, 0.9)
    )
    bce_loss = nn.BCELoss()

    for step in tqdm(range(1, args.max_epochs + 1)):

        # Critic/discriminator f optimization
        unfreeze(disc_f)
        freeze(gen_T)
        f_stats_dict = None
        for i_c in range(args.disc_iters):
            disc_f_opt.zero_grad()

            # Sample Y from the target distribution
            t_y_batch = train_y_sampler.sample(args.batch_size)
            Y = t_y_batch["x"]

            # Sample X from the source distribution
            t_batch = train_x_sampler.sample(args.batch_size)
            simulations_budget = simulations_budget + args.batch_size
            X = t_batch["x"]
            X_params = t_batch["params"]

            # Generate samples from the generator
            T_X = sample_from_generator(
                gen_T, X, X_params, z_size=Z_SIZE, args=args
            )

            f_X = disc_f(T_X).view(-1)
            f_Y = disc_f(Y).view(-1)

            label = torch.ones(
                (f_Y.shape[0],), device=f_Y.device, dtype=torch.float
            )
            f_Y_err = bce_loss(f_Y, label)
            f_Y_err.backward()

            label = torch.zeros(
                (f_X.shape[0],), device=f_X.device, dtype=torch.float
            )
            f_X_err = bce_loss(f_X, label)
            f_X_err.backward()

            disc_loss = f_Y_err + f_X_err

            # Compute the optimization step
            disc_f_opt.step()

            f_stats_dict = track_f_stats(
                f_stats_dict,
                f_X=f_X,
                f_Y=f_Y,
                loss_value=disc_loss.item(),
                wandb=wandb,
                step=step,
                compute_mean=False,
                validation_stats=False,
            )

        f_stats_dict = track_f_stats(
            f_stats_dict,
            wandb=wandb,
            step=step,
            compute_mean=True,
            validation_stats=False,
            logger=logger,
        )

        # Generator optimization
        unfreeze(gen_T)
        freeze(disc_f)
        tr_t_losses = []
        for i_g in range(args.t_iters):
            gen_opt.zero_grad()

            # Sample from the generator
            t_batch = train_x_sampler.sample(args.batch_size)
            simulations_budget = simulations_budget + args.batch_size
            X = t_batch["x"]
            X_params = t_batch["params"]

            T_X = sample_from_generator(
                gen_T, X, X_params, z_size=Z_SIZE, args=args
            )

            label = torch.ones(
                (T_X.shape[0],), device=T_X.device, dtype=torch.float
            )

            disc_f_X = disc_f(T_X).view(-1)
            # Compute the loss
            T_loss = bce_loss(disc_f_X, label)

            # Compute the gradient
            T_loss.backward()
            # Update the generator
            gen_opt.step()

            tr_t_losses.append(T_loss.item())

        # Potential stat
        # gen_T loss
        wandb_log(wandb, {"tr_T_loss": np.mean(tr_t_losses)}, step=step)
        logger.info(
            "Step: {}, tr_T_loss: {}".format(step, np.mean(tr_t_losses))
        )

        # log simulation budget per step/epoch
        logger.debug(
            "Simulation budget: {} per step {}".format(
                simulations_budget, step
            )
        )
        wandb_log(wandb, {"simulation_budget": simulations_budget}, step=step)

        # Validation method
        c_val_loss = torch.inf
        lapsed_time = time.time() - total_time
        # check if lapsed time is more than approx 11.45 hours
        if "timeout_early_stop" in args and args["timeout_early_stop"] > 0:
            timeout = (
                True
                if lapsed_time / 3600 >= args["timeout_early_stop"]
                else False
            )
        else:
            timeout = False

        if step == 1 or step % args.val_interval == 0 or timeout:
            logger.info("Computing validation loss and metrics")
            freeze(disc_f)
            freeze(gen_T)

            f_stats_dict = None
            mean_mmd = []
            std_mmd = []

            # check if it is a numerical artificial experiment
            if (
                "pendulum" in args["data_sampler"]["name"]
                or "advdiff" in args["data_sampler"]["name"]
            ):
                params_exists = True
            else:
                params_exists = False

            params_dim = gen_T.params_dim
            if params_exists:
                score_res, model_evals = eval_numerical_cond_samples(
                    gen_T,
                    val_y_sampler,
                    params_dim,
                    Z_SIZE,
                    metrics,
                    ["marginal_score"],
                    args.val_batch_size,
                    args,
                    device=gen_T.device,
                )
                mean_mmd, std_mmd = score_res[0], score_res[1]
                preds = model_evals["preds"]
                test_sims = model_evals["test_sims"]

                if preds.dim() > 2:
                    f_X = disc_f(preds.flatten(start_dim=0, end_dim=1)).view(
                        -1
                    )
                    f_Y = disc_f(
                        test_sims.flatten(start_dim=0, end_dim=1)
                    ).view(-1)
                else:
                    f_X = disc_f(preds).view(-1)
                    f_Y = disc_f(test_sims).view(-1)

                # loss bce fake and real
                label = torch.ones(
                    (f_Y.shape[0],), device=f_Y.device, dtype=torch.float
                )
                f_Y_err = bce_loss(f_Y, label)
                label = torch.zeros(
                    (f_X.shape[0],), device=f_X.device, dtype=torch.float
                )
                f_X_err = bce_loss(f_X, label)
                disc_loss = f_Y_err + f_X_err

                f_stats_dict = track_f_stats(
                    f_stats_dict=f_stats_dict,
                    f_X=f_X,
                    f_Y=f_Y,
                    loss_value=disc_loss.item(),
                    wandb=wandb,
                    step=step,
                    compute_mean=False,
                    validation_stats=True,
                    logger=logger,
                )

                # save some plots
                # hydra output dir
                if "visualize" in args and args["visualize"]:
                    chk_dirame = os.path.dirname(out_path_chkp)
                    n_plot_samples = 10
                    if "pendulum" in args["data_sampler"]["name"]:
                        os.makedirs(f"{chk_dirame}/plots", exist_ok=True)
                        dt = args["data_sampler"]["forward_model"]["dt"]
                        t = torch.linspace(
                            0.0,
                            dt * (test_sims.shape[-1] - 1),
                            test_sims.shape[-1],
                        )
                        fig_out = plot_traj(
                            t.tolist(),
                            model_evals["params"],
                            model_evals["test_sims"],
                            model_evals["preds"],
                            None,
                            None,
                            n_plot_samples,
                            True,
                            f"{chk_dirame}/plots",
                        )
                        if step % 1000 == 0 or step == 1:
                            wandb_log(
                                wandb,
                                {f"chart": wandb.Image(fig_out)},
                                step=step,
                            )

            # calculate the mean of the validation MMD
            mean_mmd = torch.mean(mean_mmd)
            if std_mmd is None or not torch.is_tensor(std_mmd):
                std_mmd = 0.0
            else:
                std_mmd = torch.mean(std_mmd)

            wandb_log(wandb, {"val_mmd": mean_mmd}, step=step)
            wandb_log(wandb, {"val_mmd_std": std_mmd}, step=step)
            logger.info(
                f"Validation, step: {step}, avg(mmd): {mean_mmd} +- {std_mmd}"
            )
            # Best model with respect to the discriminator loss, i.e when we are not able to distinguish between the generated and the real data
            c_val_loss = mean_mmd

            if (
                step % args.save_interval == 0
                or step == args.max_epochs
                or timeout
            ):
                if abs(c_val_loss) < abs(best_loss):
                    best_loss = c_val_loss
                    save_checkpoints(
                        gen_T,
                        disc_f,
                        gen_opt,
                        disc_f_opt,
                        step,
                        out_path_chkp,
                        logger,
                        ending_name="best",
                    )

            if timeout:
                # End the training loop
                break
    # End of the training loop

    # Save the final model
    save_checkpoints(
        gen_T,
        disc_f,
        gen_opt,
        disc_f_opt,
        step,
        out_path_chkp,
        logger,
        ending_name="final",
    )

    results = {
        "T": gen_T,
        "f": disc_f,
        "simulations_budget": simulations_budget,
        "best_loss": best_loss,
    }
    logger.info(
        f"Training completed.\n\t Simulations budget: {simulations_budget} \n\t Best loss: {best_loss}"
    )

    return results
