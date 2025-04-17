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
from src.sampler.distributions import sample_latent_noise
from src.forward_models.pendulum.visualization import plot_traj
from src.metrics.mmd import MMDLoss, RBF, estimate_mmd_bandwidth
from src.train.utils import stopping_criterion

from src.metrics.ot_loss import ot_knot_dist, gradient_optimality
from src.train.gans.wgans import gradient_penalty
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


def track_f_stats(
    f_stats_dict,
    f_X=None,
    f_Y=None,
    step=None,
    wandb=None,
    compute_mean=False,
    validation_stats=False,
):
    with torch.no_grad():
        if f_stats_dict is None:
            f_stats_dict = {
                "tr_f_loss (abs)": [],
                "tr_f_X": [],
                "tr_f_Y": [],
                "tr_f_X - tr_f_Y": [],
                "tr_f_X_var": [],
                "tr_f_Y_var": [],
                "tr_f_Y_var - tr_f_X_var": [],
                "tr_f_X_max": [],
                "tr_f_Y_max": [],
                "tr_f_X_min": [],
                "tr_f_Y_min": [],
                "tr_f_X_hist": [],
                "tr_f_Y_hist": [],
            }

        if not compute_mean:
            f_X_mean = f_X.mean().item()
            f_Y_mean = f_Y.mean().item()

            f_loss_value = abs(f_X_mean - f_Y_mean)
            f_stats_dict["tr_f_loss (abs)"].append(f_loss_value)
            f_stats_dict["tr_f_X"].append(f_X_mean)
            f_stats_dict["tr_f_Y"].append(f_Y_mean)
            f_stats_dict["tr_f_X - tr_f_Y"].append(f_X_mean - f_Y_mean)
            f_stats_dict["tr_f_X_var"].append(f_X.var().item())
            f_stats_dict["tr_f_Y_var"].append(f_Y.var().item())
            f_stats_dict["tr_f_Y_var - tr_f_X_var"].append(
                (f_Y.var() - f_X.var()).item()
            )
            f_stats_dict["tr_f_X_max"].append(f_X.max().item())
            f_stats_dict["tr_f_Y_max"].append(f_Y.max().item())
            f_stats_dict["tr_f_X_min"].append(f_X.min().item())
            f_stats_dict["tr_f_Y_min"].append(f_Y.min().item())
            # track the histogram of f_Y and f_X every 50 epochs
            if step % 50 == 0 or step == 1:
                hist_f_y = np.histogram(f_Y.detach().cpu().numpy())
                hist_f_xz = np.histogram(f_X.detach().cpu().numpy())
                f_stats_dict["tr_f_X_hist"].append(hist_f_xz)
                f_stats_dict["tr_f_Y_hist"].append(hist_f_y)
        else:
            for key in f_stats_dict.keys():
                val = f_stats_dict[key]
                if val is not None and len(val) > 0:
                    if "hist" in key:
                        f_stats_dict[key] = wandb.Histogram(
                            np_histogram=f_stats_dict[key][-1]
                        )
                    else:
                        val = np.mean(f_stats_dict[key])
                        f_stats_dict[key] = val

    if wandb is not None:
        if validation_stats:
            # copy and replace keys tr_ with val_
            val_f_stats_dict = {}
            for key in f_stats_dict.keys():
                val = f_stats_dict[key]
                if not key == "tr_f_X_hist" and not key == "tr_f_Y_hist":
                    if isinstance(val, list) and len(val) > 0:
                        val = np.mean(val)
                val_f_stats_dict[key.replace("tr_", "val_")] = val

            f_stats_dict = val_f_stats_dict
        # send on wandb
        # remove attribute from f_stats_dict if empty
        for key in list(f_stats_dict.keys()):
            if (
                isinstance(f_stats_dict[key], list)
                and len(f_stats_dict[key]) == 0
            ):
                del f_stats_dict[key]

        wandb_log(wandb, f_stats_dict, step=step)

    return f_stats_dict


def knot_training(
    train_x_sampler,
    train_y_sampler,
    val_y_sampler,
    T,
    f,
    T_opt,
    f_opt,
    args,
    wandb,
    logger,
    T_scheduler=None,
    f_scheduler=None,
    out_path_chkp=None,
):
    COST = args.cost

    # Generator Noise parameters
    stoch_map = False
    if (
        ("model_name" in args["model"] and args.model["model_name"] == "not")
        or ("type" in args["model"] and args.model["type"] == "det")
        or args["z_size"] == 0
        or args["gamma"] == 0
    ):
        Z_STD = 0.0
        Z_SIZE = 1
        stoch_map = False
    else:
        Z_STD = 0.1
        Z_SIZE = (
            args.z_size
        )  # number of noise samples to associate with each input X=[BATCH_SIZE, 1, DIM] -> X_REPEAT=[BATCH_SIZE, Z_SIZE, DIM]
        stoch_map = True

    ALPHA = args.alpha
    # gamma
    gamma = args.gamma

    num_gpus = torch.cuda.device_count()
    logger.info(
        f"Starting training {'stochastic' if stoch_map else 'deterministic'} OT-Map"
    )
    logger.info(f"Selected device: {T.device} among {num_gpus} GPUs")
    logger.info(f"Training for {args.max_epochs} epochs")
    logger.info(f"BATCH_SIZE: {args.batch_size}")
    logger.info(f"Generator model: \n\t{T}")
    logger.info(
        f"Generator Optmizer: {T_opt}. \n\t Optimized parameters: {str(T_opt)}"
    )

    logger.info(f"Discriminator model: \n\t{f}")
    logger.info(
        f"Discriminator Optmizer: {f_opt}. \n\t Optimized parameters: {str(f_opt)}"
    )

    start_time = time.time()
    best_loss = torch.inf
    y_sims_budget = 0
    x_sims_budget = 0

    # estimmate MMD bandwidth
    bandwidth = estimate_mmd_bandwidth(val_y_sampler, median_heuristic=True)
    val_y_sampler.reset()
    rbf = RBF(bandwidth=bandwidth, n_kernels=6, device=val_y_sampler.device)
    if stoch_map:
        metrics = [MMDLoss(kernel=rbf)]
    else:
        metrics = ["abs_err"]
    gamma_max = args.gamma
    gamma_min = 0.0
    gamma_iters = 5000

    for step in tqdm(range(1, args.max_epochs + 1)):
        if "gamma_incr" in args and args["gamma_incr"]:
            gamma = min(
                gamma_max,
                gamma_min + (gamma_max - gamma_min) * step / gamma_iters,
            )

        one_epoch_time = time.time()

        # T Generator optimization
        unfreeze(T)
        freeze(f)
        for t_iter in range(args.t_iters):
            T_opt.zero_grad()
            with torch.no_grad():
                t_batch = train_x_sampler.sample(args.batch_size)
            X = t_batch["x"]

            nrows = X.shape[0]
            x_dim = X.shape[1:]
            X_params = t_batch["params"]

            x_sims_budget += X.shape[0]

            # Add noise to the input
            if stoch_map:
                X = X.reshape((-1, 1) + x_dim).repeat(
                    [1, Z_SIZE] + [1] * len(x_dim)
                )
                X_params = X_params.reshape(
                    (-1, 1) + X_params.shape[1:]
                ).repeat([1, Z_SIZE] + [1] * len(X_params.shape[1:]))

                with torch.no_grad():
                    Z = sample_latent_noise(
                        n_samples=X.shape[0],
                        z_dim=T.z_dim,
                        z_size=Z_SIZE,
                        type=args["z_dist"] if "z_dist" in args else "gauss",
                        device=X.device,
                    )

                    if len(x_dim) > 1:
                        extend_dim = tuple(([1] * len(tuple(x_dim[:-1]))))
                        Z = Z.reshape((-1, Z_SIZE) + extend_dim + (T.z_dim,))
                        Z = Z.repeat((1, 1) + tuple(x_dim[:-1]) + (1,))
                    X_input = torch.cat([X, Z], dim=-1).flatten(
                        start_dim=0, end_dim=1
                    )
                    X_params = X_params.flatten(start_dim=0, end_dim=1)
            else:
                X_input = X

            if args["model"]["nnet_arch"] == "odenet":
                T_X = T.simulate(
                    X_input,
                    context=X_params,
                )
            else:
                T_X = T(
                    X_input,
                    context=X_params,
                )

            # Compute the loss
            if COST == "energy-ang-to-euc":
                pred = torch.cat([torch.cos(T_X), torch.sin(T_X)], dim=1)
            else:
                pred = T_X

            f_X = f(pred)

            if stoch_map:
                # Reshape T_X to [nrows, Z_SIZE, -1] for variance computation in ot_knot_dist
                T_X = T_X.reshape((nrows, Z_SIZE) + (-1,))

            knot_dist, c_cost, T_var = ot_knot_dist(
                T_X,
                X,
                gamma,
                COST,
                ALPHA,
                Z_SIZE,
            )
            T_loss = knot_dist - f_X.mean()
            T_loss.backward()
            if "clip_grad" in args and args["clip_grad"][0]:
                torch.nn.utils.clip_grad_value_(
                    T.parameters(), args["clip_grad"][1]
                )  # or torch.nn.utils.clip_grad_norm_

            T_opt.step()

        if T_scheduler is not None:
            T_scheduler.step()

        # f(T() stats
        f_TX_mean = f_X.mean().item()
        f_TX_var = f_X.var().item()

        # T distance
        wandb_log(wandb, {"tr_T_knot_dist": knot_dist.item()}, step=step)
        logger.debug(
            f"Training, step: {step}, T knot distance: {knot_dist.item()}"
        )
        # cost distance
        wandb_log(wandb, {"tr_c_cost": c_cost.item()}, step=step)

        wandb_log(
            wandb,
            {"tr_ratio_fX/cost": f_TX_mean / knot_dist.item()},
            step=step,
        )

        # Variance of T_X
        if T_var > 0:
            wandb_log(wandb, {"tr_T_var": T_var.item()}, step=step)
            logger.debug(f"Training, step: {step}, T variance: {T_var.item()}")

            logger.debug(
                f"Training, step: {step}, T cost: {knot_dist.item() + gamma * T_var.item()}"
            )

        # Potential stat
        wandb_log(wandb, {"tr_T_f_x": f_TX_mean}, step=step)
        logger.debug(f"Training, step: {step}, f_X: {f_TX_mean}")
        wandb_log(wandb, {"tr_T_f_var": f_TX_var}, step=step)
        logger.debug(f"Training, step: {step}, f_X variance: {f_TX_var}")

        # T loss
        wandb_log(wandb, {"tr_T_loss": T_loss.item()}, step=step)
        logger.debug(f"Training, step: {step}, T loss: {T_loss.item()}")

        T_loss_value = T_loss.item()

        # garbage_time = time.time()
        del X, X_params, X_input, pred, T_X
        if stoch_map:
            del Z

        # f optimization
        freeze(T)
        unfreeze(f)

        disc_iters = (
            args["disc_iters"]
            if "disc_iters" in args and args["disc_iters"] > 0
            else 1
        )
        if "disc_increase" in args and args["disc_increase"][0]:
            disc_iters = (
                disc_iters * args["disc_increase"][2]
                if step % args["disc_increase"][1] == 0
                else disc_iters
            )
            if step % args["disc_increase"][1] == 0:
                logger.info(
                    f"Increasing the number of discriminator iterations to {disc_iters} at step {step}"
                )

        f_stats = None
        for f_iter in range(disc_iters):
            f_opt.zero_grad()
            with torch.no_grad():
                t_batch = train_x_sampler.sample(args.batch_size)
            X = t_batch["x"]
            x_dim = X.shape[1:]
            nrows = X.shape[0]
            X_params = t_batch["params"]
            x_sims_budget += X.shape[0]

            t_y_batch = train_y_sampler.sample(args.batch_size)
            Y = t_y_batch["x"]
            y_sims_budget += Y.shape[0]

            with torch.no_grad():
                if stoch_map:
                    # Add noise to the input
                    X = X.reshape((-1, 1) + x_dim).repeat(
                        [1, Z_SIZE] + [1] * len(x_dim)
                    )
                    X_params = X_params.reshape(
                        (-1, 1) + X_params.shape[1:]
                    ).repeat([1, Z_SIZE] + [1] * len(X_params.shape[1:]))
                    Z = sample_latent_noise(
                        n_samples=X.shape[0],
                        z_dim=T.z_dim,
                        z_size=Z_SIZE,
                        type=args["z_dist"] if "z_dist" in args else "gauss",
                        device=X.device,
                    )
                    if len(x_dim) > 1:
                        extend_dim = tuple(([1] * len(tuple(x_dim[:-1]))))
                        Z = Z.reshape((-1, Z_SIZE) + extend_dim + (T.z_dim,))
                        Z = Z.repeat((1, 1) + tuple(x_dim[:-1]) + (1,))
                    X_input = torch.cat([X, Z], dim=-1).flatten(
                        start_dim=0, end_dim=1
                    )
                    X_params = X_params.flatten(start_dim=0, end_dim=1)
                else:
                    X_input = X

                if args["model"]["nnet_arch"] == "odenet":
                    T_X = T.simulate(X_input, context=X_params)
                else:
                    T_X = T(X_input, context=X_params)

            # Compute discriminator Loss
            if COST == "energy-ang-to-euc":
                pred = torch.cat([torch.cos(T_X), torch.sin(T_X)], dim=-1)
                f_X = f(pred)
                f_Y = f(torch.cat([torch.cos(Y), torch.sin(Y)], dim=-1))
            else:
                f_X = f(T_X)
                f_Y = f(Y)

            f_loss = f_X.mean() - f_Y.mean()

            # Add the gradient optimality cost if needed
            f_loss_value = abs(f_loss.item())
            lambda_gopt = 0
            if "grad_penalty" in args and args["grad_penalty"] > 0:
                lambda_gopt = args["grad_penalty"]
                if args["grad_regularizer"] == "w2l1":
                    go_loss = gradient_penalty(f=f, T_X=T_X, Y=Y)
                elif args["grad_regularizer"] == "gopt":
                    go_loss = gradient_optimality(f=f, T_X=T_X, X=X)
                else:
                    raise ValueError("Invalid gradient regularizer")

                go_loss = lambda_gopt * go_loss
                f_loss = f_loss + go_loss
                f_gop_val = go_loss.item()

            # backward_time = time.time()
            f_loss.backward()
            f_opt.step()
            # logger.debug(
            #    f"Discriminator backward pass time compute: {time.time() - backward_time}"
            # )
            if f_scheduler is not None:
                f_scheduler.step()

            # f stats
            f_stats = track_f_stats(
                f_stats, f_X, f_Y, step=step, compute_mean=False
            )

        # Average f stats and send to wandb
        f_stats = track_f_stats(
            f_stats, step=step, wandb=wandb, compute_mean=True
        )

        if lambda_gopt > 0:
            wandb_log(wandb, {f"tr_gop_loss": f_gop_val}, step=step)
            wandb_log(
                wandb, {f"tr_f_gop_loss": f_loss_value + f_gop_val}, step=step
            )

        wandb_log(
            wandb,
            {"tr W_2= T_loss + f_loss": T_loss_value + f_loss_value},
            step=step,
        )

        logger.debug(f"Training, step: {step}, f loss: {f_loss_value}")
        logger.debug(
            f"Training, step: {step}, W_2= T_loss + f_loss: {T_loss_value + f_loss_value}"
        )

        # log simulation budget per step/epoch
        logger.debug(
            "Simulation budget: {} per step {}".format(y_sims_budget, step)
        )
        # simulation budget
        wandb_log(wandb, {"y_sims_budget": y_sims_budget}, step=step)
        wandb_log(wandb, {"x_sims_budget": x_sims_budget}, step=step)

        # Garbage collection
        del f_loss, Y, X, T_X, X_input
        if stoch_map:
            del Z

        # Validation
        c_val_loss = torch.inf

        # Check Early stopping
        early_stop = stopping_criterion(
            args, x_sims_budget, y_sims_budget, start_time, logger
        )

        # Validation method
        if (
            step == 1
            or step % args.val_interval == 0
            or early_stop
            or step == args.max_epochs
        ):
            logger.info("Computing validation loss and metrics")
            freeze(f)
            freeze(T)
            # val_w2_losses = []
            mean_mmd = []
            std_mmd = []

            f_stats_dict = None

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
                    T,
                    val_y_sampler,
                    params_dim,
                    Z_SIZE,
                    metrics,
                    ["marginal_score"],
                    args.val_batch_size,
                    args,
                    device=T.device,
                )
                mean_mmd = score_res[0]
                std_mmd = score_res[1] if len(score_res) > 1 else None
                preds = model_evals["preds"]
                test_sims = model_evals["test_sims"]
                # Compute Generator Lossx_dim = X.shape[1:]
                # Repeat for noise samples
                # x_dim = X_sims.shape[-1]
                # sims = X.reshape((-1, 1) + (x_dim)).repeat(1, z_samples, 1, 1)
                # knot_dist, c_cost, T_var = ot_knot_dist(
                #    T_X, X, gamma, COST, ALPHA, Z_SIZE
                # )
                # Compute discriminator Loss
                if COST == "energy-ang-to-euc":
                    preds = torch.cat(
                        [torch.cos(preds), torch.sin(preds)], dim=1
                    )
                    Y_input = test_sims.flatten(start_dim=0, end_dim=1)
                    f_X = f(preds.flatten(start_dim=0, end_dim=1))
                    f_Y = f(
                        torch.cat(
                            [torch.cos(Y_input), torch.sin(Y_input)], dim=1
                        )
                    )
                else:
                    f_X = []
                    f_Y = []
                    if stoch_map and preds.dim() > 2:
                        # predict by batches of args.batch_size
                        for i in range(0, preds.shape[0], args.val_batch_size):
                            preds_batch = preds[i : i + args.batch_size]
                            test_sims_batch = test_sims[
                                i : i + args.batch_size
                            ]
                            f_X_batch = f(
                                preds_batch.flatten(start_dim=0, end_dim=1)
                            )
                            f_Y_batch = f(
                                test_sims_batch.flatten(start_dim=0, end_dim=1)
                            )
                            f_X.append(f_X_batch)
                            f_Y.append(f_Y_batch)
                        f_X = torch.cat(f_X)
                        f_Y = torch.cat(f_Y)
                    else:
                        f_X = f(preds)
                        f_Y = f(test_sims)

                f_loss = f_X.mean() - f_Y.mean()
                f_stats_dict = track_f_stats(
                    f_stats_dict,
                    f_X=f_X,
                    f_Y=f_Y,
                    step=step,
                    compute_mean=False,
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

                del f_X, f_Y, preds, test_sims, model_evals
            else:
                raise ValueError("Not implemented yet")

            # calculate the mean of the validation losses
            f_stats_dict = track_f_stats(
                f_stats_dict,
                step=step,
                wandb=wandb,
                compute_mean=True,
                validation_stats=True,
            )

            # mmd
            mean_mmd = torch.mean(mean_mmd)
            if std_mmd is None or not torch.is_tensor(std_mmd):
                std_mmd = 0.0
            else:
                std_mmd = torch.mean(std_mmd)
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
                        T,
                        f,
                        T_opt,
                        f_opt,
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
        T,
        f,
        T_opt,
        f_opt,
        step,
        out_path_chkp,
        logger,
        ending_name="final",
    )

    results = {
        "T": T,
        "f": f,
        "y_sims_budget": y_sims_budget,
        "best_loss": best_loss,
    }
    logger.info(
        f"Training completed.\n\t Simulations budget: {y_sims_budget} \n\t Best loss: {best_loss}"
    )

    return results
