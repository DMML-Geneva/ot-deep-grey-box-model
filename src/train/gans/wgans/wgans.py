import torch
from torch import nn


EPOCHS = 500
N_CRITIC = 1
CLIP = 0.01  # 1e-3

import numpy as np

import torch.utils.data
import torch
from torch import autograd

import gc
import os

from src.train.ot_physics.utils import (
    unfreeze,
    freeze,
    wandb_log,
    save_checkpoints,
)

from src.evaluation.evaluator import (
    eval_marginal_lik_samples,
    eval_marginal_samples,
    eval_numerical_cond_samples,
)

from src.sampler.distributions import sample_latent_noise
from src.forward_models.pendulum.visualization import plot_traj
from src.metrics.mmd import MMDLoss, RBF, estimate_mmd_bandwidth
from src.train.utils import stopping_criterion
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


def gradient_penalty(f, T_X, Y):
    """W2GAN gradient penalty by following: https://github.com/igul222/improved_wgan_training/"""
    if T_X.shape[0] != Y.shape[0]:
        with torch.no_grad():
            min_batch = np.minimum(T_X.shape[0], Y.shape[0])
            # random shuffle
            perm = torch.randperm(T_X.shape[0])
            Y_hat = T_X[perm[:min_batch]]
            Y = Y[:min_batch] if Y.shape[0] < min_batch else Y
    else:
        Y_hat = T_X

    Y_hat = Y_hat.reshape_as(Y)

    # uniform random between 0 and 1
    alpha_dim = (Y_hat.shape[0],) + tuple([1] * (Y_hat.ndim - 1))
    alpha = torch.rand(alpha_dim, device=Y_hat.device)
    diff = Y_hat - Y
    interpolates = Y + alpha * diff
    if interpolates.requires_grad is False:
        interpolates.requires_grad = True
    f_interpolates = f(interpolates)

    gradients = autograd.grad(
        outputs=f_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(f_interpolates.size(), device=Y_hat.device),
        create_graph=True,
        retain_graph=True,
    )[0]
    slopes = torch.linalg.norm(gradients, ord=2, dim=-1)
    penalty = ((slopes - 1) ** 2).mean()
    return penalty


def sample_from_generator(gen_T, X, X_params, z_size=6, args=None):
    x_dim = X.shape[1:]

    if gen_T.z_dim > 0:
        with torch.no_grad():
            X = X.reshape((-1, 1) + x_dim).repeat(
                [1, z_size] + [1] * len(x_dim)
            )
            X_params = X_params.reshape((-1, 1) + X_params.shape[1:]).repeat(
                [1, z_size] + [1] * len(X_params.shape[1:])
            )
            X_params = X_params.flatten(start_dim=0, end_dim=1)

            Z = sample_latent_noise(
                n_samples=X.shape[0],
                z_dim=gen_T.z_dim,
                z_size=z_size,
                type=args["z_dist"] if "z_dist" in args else "gauss",
                device=X.device,
            )
            if len(x_dim) > 1:
                extend_dim = tuple(([1] * len(tuple(x_dim[:-1]))))
                Z = Z.reshape((-1, z_size) + extend_dim + (gen_T.z_dim,))
                Z = Z.repeat((1, 1) + tuple(x_dim[:-1]) + (1,))
            X_input = torch.cat([X, Z], dim=-1).flatten(start_dim=0, end_dim=1)

    else:
        X_input = X

    if args["model"]["nnet_arch"] == "odenet":
        preds = gen_T.simulate(X_input, context=X_params)
    else:
        preds = gen_T(X_input, context=X_params)

    return preds


def train_wgans(
    train_x_sampler,
    train_y_sampler,
    val_y_sampler,
    gen_T,
    disc_f,
    args,
    wandb,
    logger,
    out_path_chkp=None,
    lamb=10.0,
):
    # Generator Noise parameters
    stoch_map = True if gen_T.z_dim > 0 else False

    Z_SIZE = (
        args.z_size
    )  # number of noise samples to associate with each input X=[BATCH_SIZE, 1, DIM] -> X_REPEAT=[BATCH_SIZE, Z_SIZE, DIM]

    num_gpus = torch.cuda.device_count()
    logger.info(
        f"Starting training W2GAN for {'stochastic' if stoch_map else 'deterministic'} tasks"
    )
    logger.info(
        f"Selected device: {train_x_sampler.device} among {num_gpus} GPUs"
    )
    logger.info(f"Generator model: \n\t{gen_T}")
    logger.info(f"Discriminator model: \n\t{disc_f}")

    logger.info(f"Training for {args.max_epochs} epochs")
    logger.info(f"BATCH_SIZE: {args.batch_size}")

    black_box = (
        args["model"]["phys_model"] is None
        or args["model"]["phys_model"] == ""
        or args["model"]["phys_model"].lower() == "none"
        or "black" in args["model"]["phys_model"].lower()
    )

    start_time = time.time()
    best_loss = torch.inf
    x_sims_budget, y_sims_budget = 0, 0

    # estimmate MMD bandwidth
    bandwidth = estimate_mmd_bandwidth(val_y_sampler, median_heuristic=True)
    val_y_sampler.reset()
    rbf = RBF(bandwidth=bandwidth, n_kernels=6, device=val_y_sampler.device)
    if stoch_map:
        metrics = [MMDLoss(kernel=rbf)]
    else:
        metrics = ["abs_err"]

    # Optimizers
    gen_opt = torch.optim.Adam(gen_T.parameters(), lr=0.0001, betas=(0.0, 0.9))
    disc_f_opt = torch.optim.Adam(
        disc_f.parameters(), lr=0.0001, betas=(0.0, 0.9)
    )

    for step in tqdm(range(1, args.max_epochs + 1)):

        # Critic/discriminator f optimization
        unfreeze(disc_f)
        freeze(gen_T)

        if "disc_warmup" in args and args["disc_warmup"]:
            disc_iters = (
                (20 * args.disc_iters)
                if ((step < 25) or (step % 500 == 0))
                else args.disc_iters
            )
        else:
            disc_iters = args.disc_iters

        for i_c in range(disc_iters):
            disc_f_opt.zero_grad()

            # sample Y from the target distribution
            with torch.no_grad():
                t_y_batch = train_y_sampler.sample(args.batch_size)
            Y = t_y_batch["x"]
            y_sims_budget += Y.shape[0]

            # Sample X from the source distribution
            with torch.no_grad():
                t_batch = train_x_sampler.sample(args.batch_size)
            if not black_box:
                X = t_batch["x"]
            else:
                X = t_batch["init_conds"]

            X_params = t_batch["params"]
            x_sims_budget += X.shape[0]

            # Sample from the generator
            T_X = sample_from_generator(
                gen_T, X, X_params, z_size=Z_SIZE, args=args
            )

            f_X = disc_f(T_X)
            f_Y = disc_f(Y)

            f_Y_mean, f_Y_var, f_Y_max, f_Y_min = (
                f_Y.mean(),
                f_Y.var(),
                f_Y.max(),
                f_Y.min(),
            )
            f_X_mean, f_X_var, f_X_max, f_X_min = (
                f_X.mean(),
                f_X.var(),
                f_X.max(),
                f_X.min(),
            )

            disc_f_loss = f_X_mean - f_Y_mean

            # Discriminator loss
            disc_f_loss_value = disc_f_loss.item()

            # gradient penalty:
            if "grad_penalty" in args and args["grad_penalty"] > 0:
                lamb = args["grad_penalty"]
                gradient_penalty_loss = gradient_penalty(
                    f=disc_f, T_X=T_X, Y=Y
                )
                disc_f_loss += lamb * gradient_penalty_loss

            # Backward pass
            disc_f_loss.backward()
            disc_f_opt.step()

            if not ("grad_penalty" in args) or args["grad_penalty"] == 0:
                for w in disc_f.parameters():
                    w.data.clamp_(-CLIP, CLIP)

        # Wandb logging

        wandb_log(
            wandb, {f"tr_f_loss (abs)": np.abs(disc_f_loss_value)}, step=step
        )
        wandb_log(wandb, {f"tr_f_loss": disc_f_loss_value}, step=step)
        logger.debug(
            f"Training, step: {step}, f(discriminator) loss : {np.abs(disc_f_loss.item())}"
        )
        wandb_log(wandb, {f"tr_f_X": f_X.mean()}, step=step)
        wandb_log(wandb, {f"tr_f_Y": f_Y.mean()}, step=step)
        wandb_log(wandb, {f"tr_f_X_var": f_X_var}, step=step)
        wandb_log(wandb, {f"tr_f_Y_var": f_Y_var.item()}, step=step)
        wandb_log(
            wandb,
            {f"tr_f_Y_var - tr_f_X_var": f_Y.var() - f_X.var()},
            step=step,
        )

        wandb_log(wandb, {f"tr_f_X_max": f_X_max.item()}, step=step)
        wandb_log(wandb, {f"tr_f_Y_max": f_Y_max.item()}, step=step)

        wandb_log(wandb, {f"tr_f_X_min": f_X_min.item()}, step=step)
        wandb_log(wandb, {f"tr_f_Y_min": f_Y_min.item()}, step=step)

        # track histograms of f_X and f_Y (discriminator outputs)
        with torch.no_grad():
            if step % 50 == 0:
                hist_f_y = np.histogram(f_Y.detach().cpu().numpy())
                hist_f_xz = np.histogram(f_X.detach().cpu().numpy())
                wandb_log(
                    wandb,
                    {
                        f"tr_f_X_hist": wandb.Histogram(
                            np_histogram=hist_f_xz
                        ),
                        f"tr_f_Y_hist": wandb.Histogram(np_histogram=hist_f_y),
                    },
                    step=step,
                )

        # Generator optimization
        unfreeze(gen_T)
        freeze(disc_f)

        gen_opt.zero_grad()
        # Sample from the generator
        with torch.no_grad():
            t_batch = train_x_sampler.sample(args.batch_size)
        if not black_box:
            X = t_batch["x"]
        else:
            X = t_batch["init_conds"]
        X_params = t_batch["params"]
        x_sims_budget += X.shape[0]

        T_X = sample_from_generator(
            gen_T, X, X_params, z_size=Z_SIZE, args=args
        )

        disc_f_X = disc_f(T_X)
        # Compute the loss
        T_loss = -disc_f_X.mean()

        # Compute the gradient
        T_loss.backward()
        # Clip the gradient of the generator.
        if "clip_grad" in args and args["clip_grad"][0]:
            torch.nn.utils.clip_grad_norm_(
                gen_T.parameters(), args["clip_grad"][1]
            )
        # Update the generator
        gen_opt.step()

        # Potential stat
        # gen_T loss
        wandb_log(wandb, {"tr_T_loss": T_loss.item()}, step=step)
        logger.debug(
            f"Training, step: {step}, gen_T loss (abs): {np.abs(T_loss.item())}"
        )
        wandb_log(wandb, {"tr_T_f_var": disc_f_X.var().item()}, step=step)
        wandb_log(wandb, {"tr_T_f_max": disc_f_X.max().item()}, step=step)
        wandb_log(wandb, {"tr_T_f_min": disc_f_X.min().item()}, step=step)

        wandb_log(
            wandb,
            {"tr W_2= T_loss + f_loss": T_loss.item() + disc_f_loss_value},
            step=step,
        )

        wandb_log(wandb, {"y_sims_budget": y_sims_budget}, step=step)
        wandb_log(wandb, {"x_sims_budget": x_sims_budget}, step=step)

        # Validation method
        c_val_loss = torch.inf

        early_stop = stopping_criterion(
            args, x_sims_budget, y_sims_budget, start_time, logger
        )

        if step == 1 or step % args.val_interval == 0 or early_stop:
            logger.info("Computing validation loss and metrics")
            freeze(disc_f)
            freeze(gen_T)
            val_f_metrics = {"mean": [], "var": [], "max": [], "min": []}
            # val_w2_losses = []
            mean_mmd = []
            std_mmd = []

            # check if it is a numerical artificial experiment
            # check if it is a numerical artificial experiment
            if (
                "pendulum" in args["data_sampler"]["name"]
                or "advdiff" in args["data_sampler"]["name"]
                or "reactdiff" in args["data_sampler"]["name"]
            ):
                params_exists = True
            else:
                params_exists = False

            if params_exists:
                params_dim = args["data_sampler"]["phys_params_sampler"]["x"][
                    "params_dim"
                ]
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
                mean_mmd = score_res[0]
                std_mmd = score_res[1] if len(score_res) > 1 else None
                preds = model_evals["preds"]
                test_sims = model_evals["test_sims"]
                f_X = []
                f_Y = []
                if stoch_map and preds.dim() > 2:
                    for i in range(0, preds.shape[0], args.val_batch_size):
                        preds_batch = preds[i : i + args.batch_size]
                        test_sims_batch = test_sims[i : i + args.batch_size]
                        f_X_batch = disc_f(
                            preds_batch.flatten(start_dim=0, end_dim=1)
                        )
                        f_Y_batch = disc_f(
                            test_sims_batch.flatten(start_dim=0, end_dim=1)
                        )
                        f_X.append(f_X_batch)
                        f_Y.append(f_Y_batch)
                    f_X = torch.cat(f_X)
                    f_Y = torch.cat(f_Y)
                else:
                    f_X = disc_f(preds)
                    f_Y = disc_f(test_sims)

                f_X_mean, f_X_var, f_X_max, f_X_min = (
                    f_X.mean(),
                    f_X.var(),
                    f_X.max(),
                    f_X.min(),
                )
                f_Y_mean, f_Y_var, f_Y_max, f_Y_min = (
                    f_Y.mean(),
                    f_Y.var(),
                    f_Y.max(),
                    f_Y.min(),
                )

                val_f_loss = f_X_mean - f_Y_mean

                val_f_metrics["mean"].append(disc_f_loss.item())
                val_f_metrics["var"].append(f_X_var.var() - f_Y_var.var())
                val_f_metrics["max"].append(f_X_max - f_Y_min)
                val_f_metrics["min"].append(f_X_min - f_Y_min)

                val_f_metrics["hist_f_x"] = np.histogram(
                    f_X.detach().cpu().numpy()
                )
                val_f_metrics["hist_f_y"] = np.histogram(
                    f_Y.detach().cpu().numpy()
                )
                wandb_log(
                    wandb,
                    {
                        f"val_f_X_hist": wandb.Histogram(
                            np_histogram=val_f_metrics["hist_f_x"]
                        ),
                        f"val_f_Y_hist": wandb.Histogram(
                            np_histogram=val_f_metrics["hist_f_y"]
                        ),
                    },
                    step=step,
                )

                # save some plots
                # hydra output dir
                if "visualize" in args and args["visualize"]:
                    chk_dirame = os.path.dirname(out_path_chkp)
                    n_plot_samples = 10
                    if "pendulum" in args["data_sampler"]["name"]:
                        os.makedirs(f"{chk_dirame}/plots", exist_ok=True)
                        dt = args["data_sampler"]["forward_model"]["dt"]
                        f_evals = args["data_sampler"]["forward_model"][
                            "f_evals"
                        ]
                        t_intg = torch.linspace(
                            0.0, dt * (f_evals - 1), f_evals
                        ).tolist()
                        fig_out = plot_traj(
                            t_intg,
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

            # calculate the mean of the validation losses
            val_f_loss = np.mean(val_f_metrics["mean"])
            wandb_log(wandb, {"val_f_loss": val_f_loss}, step=step)
            wandb_log(
                wandb, {"val_f_loss (abs)": np.abs(val_f_loss)}, step=step
            )
            logger.info(f"Validation, step: {step}, f loss: {val_f_loss}")

            # calculate the mean of the validation MMD
            mean_mmd = torch.mean(mean_mmd)
            if std_mmd is None or not torch.is_tensor(std_mmd):
                std_mmd = 0.0
            else:
                std_mmd = torch.mean(std_mmd)
            # log the validation MMD or the absolute error
            if stoch_map:
                wandb_log(wandb, {"val_mmd": mean_mmd}, step=step)
                wandb_log(wandb, {"val_mmd_std": std_mmd}, step=step)
                logger.info(
                    f"Validation, step: {step}, avg(mmd): {mean_mmd} +- {std_mmd}"
                )
            else:
                wandb_log(wandb, {"val_abs_err": mean_mmd}, step=step)
                wandb_log(wandb, {"val_abs_err_std": std_mmd}, step=step)
                logger.info(
                    f"Validation, step: {step}, val_abs_err: {mean_mmd} +- {std_mmd}"
                )
            # Best model with respect to the discriminator loss, i.e when we are not able to distinguish between the generated and the real data
            c_val_loss = mean_mmd

            if (
                step % args.save_interval == 0
                or step == args.max_epochs
                or early_stop
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

            if early_stop:
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
        "y_sims_budget": y_sims_budget,
        "best_loss": best_loss,
    }
    logger.info(
        f"Training completed.\n\t Simulations budget: {x_sims_budget} \n\t Best loss: {best_loss}"
    )

    return results
