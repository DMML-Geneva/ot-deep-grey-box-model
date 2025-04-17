import os
import logging
import argparse
import yaml
import time

import torch

from matplotlib import pyplot as plt
import numpy as np

from src.forward_models.pendulum.visualization import (
    subplot_trajectories,
)
from src.nnets.utils import load_model

from src.metrics.mse import rmse, relative_rmse, nrmse, abs_err
from src.metrics.mmd import MMDLoss, RBF, estimate_mmd_bandwidth
from src.metrics.c2st_torch import c2st

from src.data_loader.config_data_loader import SimulatorDataset
from src.data_loader.reactdiff_dataset import ReactionDiffusionDataset
from src.data_loader.config_data_loader import (
    config_phys_sampler,
    get_x_params_sampler,
)
from src.forward_models.init_physics_model import init_physics_solver_model
from src.data_loader.data_loader import get_data_loader
from src.io.utils import find_files
from src.train.ot_physics.utils import (
    unfreeze,
    freeze,
    wandb_log,
)

sort_list_by_indexes = lambda lst, indexes: [lst[i] for i in indexes]
append_list = lambda lst, lst_2: [lst.append(i) for i in lst_2]

import torch
from torch import nn


def lex_sort(inc_params, inc_sim, c_params, c_sim):
    # Get the indices that would sort the array by the first two columns
    indices = np.lexsort(
        (inc_params.cpu().numpy(), inc_sim[:, 0].cpu().numpy())
    )

    # Sort the arrays using the computed indices
    inc_params = inc_params[indices]
    inc_sim = inc_sim[indices]

    # Get the indices that would sort the array by the first two columns
    indices = np.lexsort(
        (c_params[:, 0].cpu().numpy(), c_sim[:, 0].cpu().numpy())
    )

    # Sort the arrays using the computed indices
    c_params = c_params[indices]
    c_sim = c_sim[indices]
    return inc_params, inc_sim, c_params, c_sim


def eval_numerical_cond_samples(
    T_model,
    test_loader,
    params_dim,
    z_size,
    metrics,
    type_evals,
    max_x_batch_size,
    config,
    device,
    with_x_sims=False,
):

    model_evals = {
        "params": [],
        "x_sims": [],
        "test_sims": [],
        "preds": [],
    }
    black_box = (
        config["model"]["phys_model"] is None
        or config["model"]["phys_model"] == ""
        or config["model"]["phys_model"].lower() == "none"
        or "black" in config["model"]["phys_model"].lower()
    )
    grey_box = not black_box
    # with_x_sims = grey_box and "joint_score" in type_evals and not isinstance(metrics[0],MMDLoss) and not metrics[0] == "c2st"
    # Retrieve all the predictions for the test set
    for i in range(len(test_loader)):
        test_samples = test_loader.sample()
        test_params = test_samples["params"]
        test_sims = test_samples["x"]
        test_init_conds = test_samples["init_conds"]
        # Retrieve incomplete physics model solver, for sampling the incomplete trajectories
        if grey_box:
            with torch.no_grad():
                phys_ic_model = init_physics_solver_model(
                    config=config, device=device
                )

        one_to_many_flag = test_params.ndim > 2 and test_params.shape[1] > 1
        if one_to_many_flag:
            test_noisy_samples = test_params.shape[
                1
            ]  # number of stochastic samples
            x_dim = test_sims.shape[2:]
        else:
            test_params = test_params.squeeze(1)
            test_sims = test_sims.squeeze(1)
            test_init_conds = test_init_conds.squeeze(1)
            test_noisy_samples = 0
            z_size = 1
            x_dim = test_sims.shape[1:]

        # Retrieve the parameters and initial conditions for the incomplete model
        if one_to_many_flag:
            if test_params.ndim > 3:
                X_params = test_params[
                    :, ::test_noisy_samples, :, :params_dim
                ].squeeze(1)

                X_init_conds = test_init_conds[
                    :, ::test_noisy_samples
                ].squeeze(1)
            else:
                X_params = test_params[
                    :, ::test_noisy_samples, :params_dim
                ].squeeze(1)

                X_init_conds = test_init_conds.reshape(
                    -1, *test_init_conds.shape[2:]
                )
                X_init_conds = X_init_conds[::test_noisy_samples]
        else:
            if test_params.ndim > 2:
                X_params = test_params.flatten(start_dim=1, end_dim=-2)
                X_params = X_params[:, :, :params_dim]
            else:
                X_params = test_params[:, :params_dim]
            X_init_conds = test_init_conds
        if grey_box:
            with torch.no_grad():
                res_sims = phys_ic_model(
                    init_conds=X_init_conds, params=X_params
                )
            X_input = res_sims["x"]
        else:
            # TODO
            X_input = X_init_conds

        # Prediction
        # T_noisy_samples = test_noisy_samples
        z_dist = config["z_dist"] if "z_dist" in config else "gauss"
        with torch.no_grad():
            pred = T_model.predict(
                X_input, context=X_params, z_samples=z_size, z_type_dist=z_dist
            )
        if one_to_many_flag:
            pred = pred.reshape(-1, z_size, *x_dim)
            if test_params.ndim > 3:
                params = test_params[:, :, :, :params_dim]
                sims = X_input.unsqueeze(1).repeat(1, test_sims.shape[1], 1, 1)
            else:
                params = test_params[:, :, :params_dim]
                sims = X_input.unsqueeze(1).repeat(
                    (1, test_sims.shape[1])
                    + tuple([1] * len(test_sims.shape[2:]))
                )
        else:
            pred = pred.reshape(-1, *x_dim)
            params = test_params[:, :params_dim]
            sims = X_input

        if len(model_evals["params"]) == 0:
            model_evals["params"] = params
            model_evals["test_sims"] = test_sims
            model_evals["preds"] = pred
            model_evals["x_sims"] = sims if with_x_sims else None
        else:
            model_evals["params"] = torch.cat(
                [model_evals["params"], params], dim=0
            )
            model_evals["test_sims"] = torch.cat(
                [model_evals["test_sims"], test_sims], dim=0
            )
            model_evals["preds"] = torch.cat(
                [model_evals["preds"], pred], dim=0
            )
            if with_x_sims:
                model_evals["x_sims"] = torch.cat(
                    [model_evals["x_sims"], sims], dim=0
                )

    if isinstance(test_loader.data_loader.dataset, ReactionDiffusionDataset):
        max_x_batch_size = (
            test_loader.data_loader.dataset.batch_size_data * max_x_batch_size
        )

    res_eval = one_to_many_evaluation(
        x_params=model_evals["params"],
        pred=model_evals["preds"],
        y_sims=model_evals["test_sims"],
        y_params=model_evals["params"],  # .clone(),
        x_sims=model_evals["x_sims"],
        metrics=metrics,
        type_evals=type_evals,
        batch_size=max_x_batch_size,
    )
    test_scores = res_eval[type_evals[0]]
    return test_scores, model_evals


def one_to_many_evaluation(
    x_params,
    pred,
    y_sims,
    y_params=None,
    x_sims=None,
    type_evals=["marginal_score", "lik_score", "joint_score"],
    metrics=["mmd", "c2st", "rmse", "rel_rmse", "nrmse", "abs_err"],
    batch_size=512,
):
    """
    Parameters
    ----------
    model: GenerativeMode

    params: torch.Tensor, shape: (n_samples, dim_params)
        It describes the known paremeters of the system
    x_sims: torch.Tensor, shape (n_samples, dim_sims)
        It describes the known states of the system
    y_params: torch.Tensor, shape: (n_samples, test_noisy_samples, dim_params)
        It describes all the paremeters of the complete system
    y_sims: torch.Tensor, shape: (n_samples, test_noisy_samples, dim_sims)
        It repreesents the states of the complete system

    Returns
    -------
    lik_scores: torch.Tensor
        It contains the likelihood scores for each fixed known y_params
    joint_scores: torch.Tensor
        It contains the joint scores for the complete system
    """
    # Check metrics
    fn_metrics = []
    if len(metrics) == 0 or "mmd" in metrics:
        mmd = MMDLoss(kernel=RBF(n_kernels=6, device=x_params.device))
        fn_metrics.append(mmd)

    if "c2st" in metrics:
        fn_metrics.append(c2st)
    if "rel_rmse" in metrics:
        fn_metrics.append(relative_rmse)
    elif "rmse" in metrics:
        fn_metrics.append(rmse)
    elif "nrmse" in metrics:
        fn_metrics.append(nrmse)
    elif "abs_err" in metrics:
        fn_metrics.append(abs_err)

    if fn_metrics == [] and len(metrics) > 0:
        fn_metrics = metrics

    # Predict/Generate Y
    # Compute likelihood score for each fixed y_params
    # Shape likelihood_score: (n_samples, len(metrics))
    lik_scores = None
    if (
        "lik_score" in type_evals
        and x_params is not None
        and not x_params.isnan().any()
    ):
        lik_scores = eval_marginal_lik_samples(
            pred,
            x_params,
            y_sims,
            metrics=fn_metrics,
        )

    # Joint samples
    joint_scores = None
    if (
        "joint_score" in type_evals
        and x_params is not None
        and not x_params.isnan().any()
        and y_params is not None
    ):
        joint_scores = eval_joint_samples(
            x_params=x_params,
            x_sims=x_sims,
            pred=pred,
            y_params=y_params,
            y_sims=y_sims,
            joint_metrics=fn_metrics,
            batch_size=batch_size,
        )

    # Marginal samples
    marginal_scores = None
    if "marginal_score" in type_evals:
        marginal_scores = eval_marginal_samples(
            pred=pred,
            y_sims=y_sims,
            metrics=fn_metrics,
            batch_size=batch_size,
        )

    return {
        "lik_score": lik_scores,
        "joint_score": joint_scores,
        "marginal_score": marginal_scores,
    }


def eval_marginal_lik_samples(
    pred,
    y_params,
    y_sims,
    metrics=[],
):
    """
    Parameters
    ----------
    model: GenerativeMode

    x_params: torch.Tensor
        It describes the known paremeters of the system
    x_sims: torch.Tensor
        It describes the known states of the system
    y_params: torch.Tensor
        It describes all the paremeters of the complete system
    y_sims: torch.Tensor
        It repreesents the states of the complete system
    """
    # Shapes summary:
    # y_params has shape (n_samples, test_noisy_samples, dim_params)
    # y_sims has shape (n_samples, test_noisy_samples, dim_sims)
    # x_params has shape (n_samples, dim_params)
    # x_sims has shape (n_samples, dim_sims)
    # pred has shape (n_samples, z_samples, dim_sims)

    # Compute the metrics for each fixed y_params
    scores = torch.full((y_params.shape[0], len(metrics)), -1.0)
    for i in range(y_params.shape[0]):
        # Compute the metrics for each fixed y_params
        for j, metric in enumerate(metrics):
            if y_sims[i].ndim > 1:
                noisy_samples = y_sims[i].shape[0]
                z_samples = pred[i].shape[0]
            else:
                noisy_samples = 1
                z_samples = 1
            score_distance = metric(
                y_sims[i].reshape(noisy_samples, -1),
                pred[i].reshape(z_samples, -1),
            )
            if isinstance(score_distance, tuple):
                score_distance = score_distance[0]
            scores[i, j] = score_distance

    return scores.mean(dim=0), scores.std(dim=0)


def eval_marginal_samples(
    pred,
    y_sims,
    batch_size=512,
    metrics=[],
):
    """
    Arguments
    ----------
    x_sims: torch.Tensor
        It describes the known states of the system
    y_sims: torch.Tensor
        It repreesents the states of the complete system
    """
    # Shapes summary:
    if abs_err in metrics:
        ndim = np.prod(pred.shape[1:])
    else:
        ndim = np.prod(pred.shape[2:])

    pred = pred.reshape(-1, ndim)
    y_sims = y_sims.reshape(-1, ndim)

    batches = int(np.ceil(len(pred) / batch_size))
    one_batch = True if c2st in metrics or abs_err in metrics else False
    if one_batch:
        scores = torch.full((len(metrics), 2), torch.inf)
        batches = 1
    else:
        scores = torch.full(
            (len(metrics), batches if batches > 0 else 1), torch.inf
        )
    j = 0
    # TODO: check evaluation criteria!
    for i in range(0, batches):
        if one_batch:
            # Single batch
            y_sims_batch = y_sims
            pred_batch = pred
        else:
            # Bathcify for MMD
            y_sims_batch = y_sims[j : j + batch_size]
            pred_batch = pred[i : i + batch_size]

        for k, metric in enumerate(metrics):
            if metric in [rmse, relative_rmse, abs_err]:
                res_metric = metric(
                    pred_batch,
                    y_sims_batch,
                )
            else:
                res_metric = metric(
                    pred_batch,
                    y_sims_batch,
                    max_batch_size=batch_size,
                )
            if isinstance(res_metric, tuple):
                score_mean = res_metric[0]
                score_std = res_metric[-1]
                score_distance = torch.tensor([score_mean, score_std])
            else:
                score_distance = res_metric

            if one_batch:
                scores[k] = score_distance
            else:
                # mmd case with multi batches
                pos = int(np.floor(i // batch_size))
                scores[k, pos] = score_distance

        j += batch_size
        if j >= len(y_sims):
            j = 0  # reset the index

    scores = scores[scores != torch.inf]
    if scores.ndim > 1 and scores.shape[1] > 1:
        return (
            scores.mean(dim=1),
            scores.std(dim=1),
        )
    else:
        return scores


def eval_joint_samples(
    x_params,
    x_sims,
    pred,
    y_params,
    y_sims,
    batch_size=512,
    joint_metrics=[],
):
    print(x_params.shape, pred.shape, y_params.shape, y_sims.shape)
    if pred.ndim < 3:
        x_params = x_params.unsqueeze(1)
        if x_sims is not None:
            print(x_sims.shape)
            x_sims = x_sims.unsqueeze(1)
        pred = pred.unsqueeze(1)
        y_params = y_params.unsqueeze(1)
        y_sims = y_sims.unsqueeze(1)

    # Flatten first the z_sample and later the d-dimensional space
    pred = pred.flatten(start_dim=0, end_dim=1).flatten(
        start_dim=1, end_dim=-1
    )
    x_params = x_params.flatten(start_dim=0, end_dim=1).flatten(
        start_dim=1, end_dim=-1
    )

    if x_sims is not None:
        x_sims = x_sims.flatten(start_dim=0, end_dim=1).flatten(
            start_dim=1, end_dim=-1
        )

    y_sims = y_sims.flatten(start_dim=0, end_dim=1).flatten(
        start_dim=1, end_dim=-1
    )
    y_params = y_params.flatten(start_dim=0, end_dim=1).flatten(
        start_dim=1, end_dim=-1
    )

    if x_sims is not None:
        X = torch.cat([pred, x_sims, x_params], dim=-1)
        Y = torch.cat([y_sims, x_sims, y_params], dim=-1)
    else:
        X = torch.cat([pred, x_params], dim=-1)
        Y = torch.cat([y_sims, y_params], dim=-1)

    scores = torch.full(
        (len(joint_metrics), 2), torch.inf
    )  # (mean, std) \times metrics

    for k, metric in enumerate(joint_metrics):
        res_metric = metric(
            X,
            Y,
            max_batch_size=batch_size,
        )
        if isinstance(res_metric, tuple):
            score_mean = res_metric[0]
            score_std = res_metric[-1]
            score_distance = torch.tensor([score_mean, score_std])
        else:
            score_distance = res_metric
        scores[k] = score_distance

    scores = scores[scores != torch.inf]
    return scores


def eval_one_to_many_models_exp(
    name,
    test_file_path,
    hash_exp_name,
    dir_models,
    params_dim,
    max_samples=-1,
    max_x_samples=-1,  # only used for max_iter foor "bootstrap" of mmd reference! and when using prior samples for marginals (no numerical exp evaluation)
    max_x_batch_size=-1,
    z_size=6,
    metrics=[],
    ref_metric_value=None,
    type_evals=["lik_score"],
    with_x_sims=False,
    seed=1,
    top_k_models=5,
    comp_avg_models_stats=False,
    output_dir="./outputs/best_models",
    device="cuda",
):
    """
    Evaluate the trained models of an one-to-many task experiment and save the best models in the output_dir.
    Parameters
    ----------
    name: str
        Name of the dataset, e.g., "pendulum"
    test_file_path: str
        Path to the Test dataset file
    hash_exp_name: str
        Hash of the experiment where the model was trained. This string will be used to find all the models with this name contained in the models_dir
    dir_models: str
        Path to the directory where the models are stored
    params_dim: int
        number of known parameters dimension of the experiment
    max_samples: int
        Maximum number of samples to evaluate
    metrics: list
        List of metrics to evaluate the model, e.g MMDLoss, RMSE, etc.
    """

    device = torch.device(device)

    # set seed
    # torch manual seed
    torch.manual_seed(seed)
    # numpy seed
    np.random.seed(seed)
    if name == "reactdiff" and z_size > 1:
        test_dataset = ReactionDiffusionDataset(
            name_dataset=name,
            data_file_path=test_file_path,
            testing_set=True,
            device=device,
        )
        max_x_batch_size = max_x_batch_size // test_dataset.batch_size_data
        print(max_x_batch_size)
    else:
        test_dataset = SimulatorDataset(
            name_dataset=name,
            data_file_path=test_file_path,
            max_train_size=max_samples,
            testing_set=True,
            device=device,
        )
    test_loader = get_data_loader(
        test_dataset,
        batch_size=max_x_batch_size if max_x_batch_size > 0 else 250,
    )  # batch size can be adjusted

    # Retrieve a sample to get the dimensions
    sample = test_loader.sample(
        1,
    )
    x_dim = sample["x"].shape[-1]
    init_cond_dim = sample["init_conds"].shape[-1]
    test_loader.reset()

    joint_flag = "joint_score" in type_evals

    score_mmd_ref = None
    mmd = None
    if metrics is None or len(metrics) == 0 or "mmd" in metrics:
        metrics = []
        # MMD metric
        ## Estimate MMD on the test set
        bandwidth = estimate_mmd_bandwidth(
            test_loader,
            median_heuristic=True,
            joint_samples=joint_flag,
            params_dim=params_dim,
        )
        rbf = RBF(bandwidth=bandwidth, n_kernels=6, device=device)
        mmd = MMDLoss(kernel=rbf)
        metrics = [mmd]
        ref_metric_value = 0.0

    test_loader.reset()
    # Compute the ground truth reference
    if mmd is not None:
        # Reference MMD marginal on the test by shuffling and splitting it in half
        score_mmd_ref = []
        # shuffle Y
        K_times = 5
        for i in range(K_times):

            n_iter_max = np.maximum(
                max_x_samples, len(test_loader) * max_x_batch_size
            )
            for i in range(0, n_iter_max, max_x_batch_size):
                test_samples = test_loader.sample()
                sims = test_samples["x"]  # n \times z \times x_dim
                sims = sims.reshape(-1, np.prod(sims.shape[2:]))
                if joint_flag:
                    params = test_samples["params"].flatten(
                        start_dim=0, end_dim=1
                    )
                    if params.ndim > 2:
                        params = params[:, :, :params_dim]
                    else:
                        params = params[:, :params_dim]
                    params = params.reshape(-1, np.prod(params.shape[1:]))
                    Y = torch.cat([sims, params], dim=-1)
                else:
                    Y = sims

                perm_Y = Y[torch.randperm(Y.shape[0])]
                Y_1, Y_2 = torch.chunk(perm_Y, 2, dim=0)
                mmd_val = mmd(
                    Y_1.reshape(-1, Y_1.shape[-1]),
                    Y_2.reshape(-1, Y_2.shape[-1]),
                )
                score_mmd_ref.append(mmd_val)

        test_loader.reset()
        score_mmd_ref = torch.tensor(score_mmd_ref)
        print(f"Reference MMD score: {score_mmd_ref.mean()}")
        print(f"Reference MMD std: {score_mmd_ref.std()}")

    # Compute reference c2st on test set
    score_c2st_ref = None
    c2st_std_ref = None
    if "c2st" in metrics and ref_metric_value is not None:
        print(f"Computing reference c2st on test set")
        score_c2st_ref = []
        all_test_set = get_data_loader(
            test_dataset,
            batch_size=(
                max_x_batch_size
                if max_x_batch_size > 0
                else 250 * len(test_loader)
            ),
        )
        if "joint_score" in type_evals:
            dataset = all_test_set.sample()
            sims = dataset["x"]  # n \times z \times x_dim
            sims = sims.reshape(-1, np.prod(sims.shape[2:]))  #

            params = dataset["params"].flatten(start_dim=0, end_dim=1)
            if params.ndim > 2:
                params = params[:, :, :params_dim]
            else:
                params = params[:, :params_dim]
            params = params.reshape(-1, np.prod(params.shape[1:]))
            Y = torch.cat([sims, params], dim=-1)
        else:
            Y = all_test_set.sample()["x"]
            Y = Y.reshape(-1, np.prod(Y.shape[2:]))

        # shuffle Y
        perm_Y = Y[torch.randperm(Y.shape[0])]
        # split by 50 and 50 percent
        l1 = int(0.5 * perm_Y.shape[0])
        Y_1, Y_2 = perm_Y[:l1], perm_Y[l1:]
        res = c2st(Y_1, Y_2, max_batch_size=max_x_batch_size)
        c2st_val, c2st_std_ref = res[0], res[-1]

        score_c2st_ref = c2st_val
        print(f"Reference c2st score: {score_c2st_ref}")
        print(f"Reference c2st std: {c2st_std_ref}")
        print()
        test_loader.reset()
    # rmse
    if "rel_rmse" in metrics:
        metrics.append(relative_rmse)
    elif "rmse" in metrics:
        metrics.append(rmse)
    elif "nrmse" in metrics:
        metrics.append(nrmse)
    elif "abs_err" in metrics:
        metrics.append(abs_err)

    file_models = find_files(dir_models, ".*" + hash_exp_name + ".*\.pt$")
    file_models.sort()
    file_models = set(file_models)

    if len(file_models) == 0:
        print(f"No model found in {dir_models}")
        return None

    models_evals = {}
    parent_dir = None

    for path_f_model in file_models:
        T_model, parent_dir, config, dic_chkpt = load_model(
            path_model_file=path_f_model,
            task_name=name,
            params_dim=params_dim,
            x_dim=x_dim,
            init_cond_dim=init_cond_dim,
            device=device,
        )

        if T_model is None:
            continue  # skip the model
        freeze(T_model)

        scores = []
        t_model_eval = time.time()
        test_loader.reset()
        test_samples = test_loader.sample()  # size=1 for testing it
        test_params = test_samples["params"]
        test_loader.reset()
        params_exists = (
            True
            if test_params is not None and not test_params.isnan().any()
            else False
        )
        if params_exists:
            score_res, _ = eval_numerical_cond_samples(
                T_model,
                test_loader,
                params_dim,
                z_size,
                metrics,
                type_evals,
                with_x_sims=with_x_sims,
                max_x_batch_size=max_x_batch_size,
                config=config,
                device=device,
            )
            if isinstance(score_res, tuple):
                scores_mean = score_res[0].item()
                scores_std = score_res[1].item()
            else:
                if len(score_res) == 1:
                    scores_mean = score_res.item()
                    scores_std = torch.nan
                else:
                    scores_mean = score_res[0].item()
                    scores_std = score_res[1].item()

            models_evals[path_f_model] = {
                "score": scores_mean,
                "std_score": scores_std,
                "model_epoch": dic_chkpt["epoch"],
            }
        else:
            # evaluate marginals from random prior samples
            for i in range(len(test_loader)):
                test_samples = test_loader.sample()
                test_params = test_samples["params"]
                test_sims = test_samples["x"]
                test_init_conds = test_samples["init_conds"]
                # Retrieve incomplete physics model solver, for sampling the incomplete trajectories
                with torch.no_grad():
                    phys_ic_model = init_physics_solver_model(
                        config=config, device=device
                    )
                # use the test init_conds samples
                # Sampler the known parameters of the incomplete model
                x_params_sampler = get_x_params_sampler(config, device=device)
                if max_x_samples > 0:
                    batch_x_size = (
                        max_x_batch_size
                        if max_x_batch_size > 0
                        and max_x_samples > max_x_batch_size
                        else max_x_samples
                    )
                else:
                    max_x_samples = test_sims.shape[0]
                    batch_x_size = test_sims.shape[0]

                preds = None
                # iterate over x batches
                for i in range(0, max_x_samples, batch_x_size):
                    X_params = x_params_sampler.sample((batch_x_size,))

                    # For numerical experiments, the init_conds are repeated. No need of repeated init_conds.
                    if len(test_init_conds.shape) > 2:
                        # numerical experiments have repeated init_conds
                        noisy_samples = test_init_conds.shape[1]
                        test_init_conds = test_init_conds.reshape(
                            -1, test_init_conds.shape[-1]
                        )[::noisy_samples]

                    init_conds = test_init_conds.clone()
                    if test_init_conds.shape[0] < X_params.shape[0]:
                        # check if init_cond have the same rows as the params otherwise repeat the init_conds

                        while init_conds.shape[0] < X_params.shape[0]:
                            diff = X_params.shape[0] - init_conds.shape[0]
                            slice = np.minimum(diff, test_init_conds.shape[0])
                            init_conds = torch.cat(
                                [init_conds, test_init_conds[:slice]],
                                dim=0,
                            )
                    elif test_init_conds.shape[0] > X_params.shape[0]:
                        # repeat x_params for missing init_conds
                        diff = test_init_conds.shape[0] - X_params.shape[0]
                        new_x_params = x_params_sampler.sample((diff,))
                        X_params = torch.cat(
                            [X_params, new_x_params[:diff]], dim=0
                        )
                    with torch.no_grad():
                        x_res_sims = phys_ic_model(
                            init_conds=init_conds, params=X_params
                        )
                    X_sims = x_res_sims["x"]
                    z_size = z_size if z_size is not None or z_size > 0 else 6
                    z_dist = (
                        config["z_dist"] if "z_dist" in config else "gauss"
                    )
                    with torch.no_grad():
                        T_XZ = T_model.predict(
                            X_sims,
                            context=X_params,
                            z_samples=z_size,
                            z_type_dist=z_dist,
                        )
                    x_dim = X_sims.shape[1:]
                    pred = T_XZ.reshape((-1, z_size) + (x_dim))
                    if preds is None:
                        preds = pred
                    else:
                        preds = torch.cat([preds, pred], dim=0)

                res_eval = one_to_many_evaluation(
                    x_params=X_params,
                    pred=preds,
                    y_sims=test_sims,
                    metrics=metrics,
                    type_evals=type_evals,
                    batch_size=batch_x_size,
                )
                test_scores = res_eval[type_evals[0]]
                scores.extend(test_scores.flatten().detach().cpu().numpy())
                del test_scores

            # average of scores
            scores = torch.tensor(scores)
            batch_score = scores.mean(dim=0).numpy()
            std_score = scores.std(dim=0).numpy()
            models_evals[path_f_model] = {
                "score": batch_score.tolist(),
                "std_score": std_score.tolist(),
                "model_epoch": dic_chkpt["epoch"],
            }

        print(
            f"Model {path_f_model} evaluated in {time.time() - t_model_eval:.4f} s"
        )

    # Select the k-best models and save
    best_scores = np.ones(top_k_models) * np.inf
    best_score_models = {}
    for path_f_model, res in models_evals.items():
        res_score = res["score"]
        if np.any(np.isinf(best_scores)):
            idx = np.where(np.isinf(best_scores))[0]
            best_scores[idx[0]] = res_score
            best_score_models[idx[0]] = (path_f_model, res_score)
        else:
            if ref_metric_value is not None:
                new_score = np.abs(res_score - ref_metric_value)
                new_best_score = np.any(
                    new_score < np.abs(best_scores - ref_metric_value)
                )
            else:
                new_best_score = np.any(res_score < best_scores)

            if new_best_score:
                # replace the worst score
                if ref_metric_value is not None:
                    idx_worst_score = np.argmax(
                        np.abs(best_scores - ref_metric_value)
                    )
                else:
                    idx_worst_score = np.argmax(best_scores)
                best_scores[idx_worst_score] = res_score
                best_score_models[idx_worst_score] = (
                    path_f_model,
                    res_score,
                    res["model_epoch"],
                )

    # Copy the best models
    if best_score_models is not None and len(best_score_models) > 0:

        os.makedirs(f"{output_dir}/{hash_exp_name}", exist_ok=True)

        for idx, best_model in best_score_models.items():
            model_epoch = -1
            if len(best_model) == 2:
                path_f_model, score = best_model
            else:
                path_f_model, score, model_epoch = best_model
            model_dir_path = os.path.dirname(path_f_model)
            salt_char = model_dir_path.split("/")[-1]
            if len(salt_char) < 8:
                # copy the whole directory
                salt_char_score = (
                    f"{salt_char}_score_{score:.4f}_epoch_{model_epoch}"
                )
                os.system(
                    f"cp -r {model_dir_path} {output_dir}/{hash_exp_name}/{salt_char_score}"
                )
            else:
                f_name = path_f_model.split("/")[-1]
                f_name = f_name.replace(
                    ".", "_score_{:.4f}_{}.".format(score, model_epoch)
                )
                os.system(
                    f"cp {path_f_model} {output_dir}/{hash_exp_name}/{f_name}"
                )
                os.system(
                    f"cp {path_f_model} {output_dir}/{hash_exp_name}/{f_name}"
                )
                model_conf_path = path_f_model.replace(
                    "_best.pt", "_config.yaml"
                )
                model_conf_path = model_conf_path.replace(
                    "_final.pt", "_config.yaml"
                )
                os.system(
                    f"cp {model_conf_path} {output_dir}/{hash_exp_name}/"
                )

        # save also all the models_evals results with stats as yaml file
        # models_evals = models_evals.detach().cpu()
        ref_metric = None
        if score_mmd_ref is not None:
            ref_metric = {
                "mean": score_mmd_ref.mean().item(),
                "std": score_mmd_ref.std().item(),
            }
        if score_c2st_ref is not None:
            ref_metric = {
                "mean": score_c2st_ref.item(),
                "std": c2st_std_ref.item(),
            }

        result = {
            "models_evals": models_evals,
            "ref_metric": ref_metric,
            "ref_y_dataset": test_file_path,
        }

        # Compute the average score of the best models and the final models
        if comp_avg_models_stats:
            # filter movel_evaluation by the keywords "best" and "final"
            best_m_evals = list(
                filter(lambda x: "best" in x, models_evals.keys())
            )
            final_m_evals = list(
                filter(lambda x: "final" in x, models_evals.keys())
            )

            # compute average score among the best models
            avg_score_best_models = np.mean(
                [models_evals[m]["score"] for m in best_m_evals]
            ).item()
            std_score_best_models = np.std(
                [models_evals[m]["score"] for m in best_m_evals]
            ).item()
            avg_score_final_models = np.mean(
                [models_evals[m]["score"] for m in final_m_evals]
            ).item()
            std_score_final_models = np.std(
                [models_evals[m]["score"] for m in final_m_evals]
            ).item()

            result["seed_avg_scores"] = {
                "seed": seed,
                "n_models": len(final_m_evals),
                "best_models": {
                    "avg_score": avg_score_best_models,
                    "std_score": std_score_best_models,
                },
                "final_models": {
                    "avg_score": avg_score_final_models,
                    "std_score": std_score_final_models,
                },
            }

        # save to file all the results
        print(result)
        yaml.dump(
            result,
            open(
                f"{output_dir}/{hash_exp_name}/{hash_exp_name}_models_evals.yaml",
                "w",
            ),
        )
        print(f"Model evaluation saved in {output_dir}")

    # TODO: DO the plotting of the best models here using the best_score_models
    return best_model


def test_predictive_check(
    task_name,
    test_file_path,
    T_model,
    phys_ic_model=None,
    max_samples=-1,
    outdir=None,
    plot_name=None,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = SimulatorDataset(
        name_dataset=task_name,
        data_file_path=test_file_path,
        max_train_size=max_samples,
        device=device,
    )
    test_loader = get_data_loader(test_dataset, batch_size=1)

    test_samples = test_loader.sample()
    test_params = test_samples["params"]
    test_sims = test_samples["x"]
    test_init_conds = test_samples["init_conds"]

    # incomplete simulations
    x_sims = None
    x_params = None
    if task_name == "pendulum":
        x_params = test_params[:, :1]
        with torch.no_grad:
            x_sims = phys_ic_model(init_conds=test_init_conds, params=x_params)

    # z_dist = config["z_dist"] if "z_dist" in config else "gauss"
    with torch.no_grad:
        pred = T_model.predict(x_sims, context=x_params, z_samples=6)

    # Predict the trajectories
    # Plot the predicted trajectories
    n_plot_samples = 16
    params = []
    sims = []
    for i in range(n_plot_samples):
        idx = np.random.randint(0, len(test_params))
        print(f"random idx {idx}")

        # incomplete trajectories
        params.extend(x_params[idx : idx + 1].tolist())
        sims.extend(x_sims[idx : idx + 1].tolist())

        # complete trajectories
        params.extend(test_params[idx : idx + 1].tolist())
        sims.extend(test_sims[idx : idx + 1].tolist())

        # TODO: change the evaluation here for one-to-many
        # test_c_sims has shape (1, 50)
        # pred has shape (1, z_samples, 50)
        # predicted params
        params.append("Pred")
        sims.extend(pred[idx : idx + 1].tolist())

    t = torch.linspace(
        0.0, phys_ic_model.dt * (test_sims.shape[1] - 1), test_sims.shape[1]
    )

    fig, axes = subplot_trajectories(
        params,
        sims,
        t,
        idx_start=0,
        n_samples=3,
        rows=8,
        cols=2,
        figsize=(16, 18),
        plot_show=False,
    )
    # plt.subplots_adjust(hspace = 1.5)

    save_dir = f"{outdir}"
    save_dir = f"{outdir}/plot_results"
    os.makedirs(f"{save_dir}", exist_ok=True)
    plt.savefig(f"{save_dir}/trajectory_preds__{plot_name}.png")
    fig.clf()


if __name__ == "__main__":

    # Argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dir_models",
        type=str,
        default="./multirun/pendulum/many_modes/wgan/gb/2025-12-27/14-26-22-wgans-pend-gb-many-modes",
    )
    parser.add_argument(
        "--test_file_path",
        type=str,
        default="./datasets/forward_models/pendulum/one_to_many/many_modes/testing/data_ab3c7ed9be53f93b31ea13deba72659a",
    )
    # parser.add_argument("--sample_size", type=int, default=1500)
    parser.add_argument(
        "--hash_exp_name",
        type=str,
        default="b0af176f7b1d439550e605188cbc1138",
    )
    parser.add_argument("--phys_model", type=str, default="pendulum")
    parser.add_argument("--params_dim", type=int, default=1)
    parser.add_argument("--top_k_models", type=int, default=5)
    parser.add_argument(
        "--type_evals", type=str, default="joint_score"
    )  # marginal_score, joint_score, lik_score
    parser.add_argument(
        "--metric", type=str, default="mmd"
    )  # c2st, mmd, abs_err
    parser.add_argument("--ref_metric_value", type=float, default=None)
    parser.add_argument("--with_x_sims", type=int, default=0)
    parser.add_argument("--comp_avg_models_stats", type=int, default=1)
    parser.add_argument("--max_x_samples", type=int, default=10000)
    parser.add_argument("--max_x_batch_size", type=int, default=5000)
    parser.add_argument("--z_size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/best_models/pendulum/many_modes/joint/mmd/wgans/seed-13/",
    )

    args = parser.parse_args()

    res_eval = eval_one_to_many_models_exp(
        name=args.phys_model,
        test_file_path=args.test_file_path,
        hash_exp_name=args.hash_exp_name,
        dir_models=args.dir_models,
        params_dim=args.params_dim,
        max_samples=-1,
        top_k_models=args.top_k_models,
        type_evals=[args.type_evals],
        metrics=[args.metric] if args.metric is not None else [],
        ref_metric_value=args.ref_metric_value,
        with_x_sims=bool(args.with_x_sims),
        comp_avg_models_stats=bool(args.comp_avg_models_stats),
        max_x_samples=args.max_x_samples,
        max_x_batch_size=args.max_x_batch_size,
        z_size=args.z_size,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
    )
