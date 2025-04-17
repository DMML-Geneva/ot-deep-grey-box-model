import json
import os

import numpy as np
from math import ceil
import torch

from src.forward_models.advdiff.advdiff_options import AdvDiffOptionParser
from src.forward_models.advdiff import AdvDiffSolver
from src.forward_models.advdiff.advdiff_model import HASHED_KEYS_NAME

from src.sampler.distributions import BoxUniformSampler
from src.utils import hash


def generate_data(
    n_samples,
    n_stoch_samples,
    range_init_mag,
    range_dcoeff,
    range_ccoeff,
    dx,
    dt,
    n_grids,
    len_episode,
    fixed_stoch_samples=False,
    noise_loc=0.0,
    noise_std=1e-3,
    batch_size=250,
    library="torch",
    method="dopri8",
    device="cpu",
    seed=1,
    outdir=None,
):

    # check inputs
    assert range_init_mag[0] <= range_init_mag[1]
    assert range_dcoeff[0] <= range_dcoeff[1]
    assert range_ccoeff[0] <= range_ccoeff[1]

    lb = torch.tensor([range_init_mag[0], range_dcoeff[0]], device=device)
    ub = torch.tensor([range_init_mag[1], range_dcoeff[1]], device=device)

    c_coeff_lb, c_coeff_ub = torch.tensor(
        [range_ccoeff[0]], device=device
    ), torch.tensor([range_ccoeff[1]], device=device)

    # Sampler
    known_params_sampler = BoxUniformSampler(lb, ub, device=device)
    samples = known_params_sampler.sample((n_samples,))
    # init conditions and boundaries
    init_conds = samples[:, 0].view(-1, 1)
    x_grid = torch.linspace(
        0.0, dx * (n_grids - 1), n_grids, device=device
    )  # 20
    init_sin = torch.sin(x_grid / x_grid[-1] * torch.pi)
    # broadcast init_conds * init_sin for boundaries conditions
    init_conds = init_conds * init_sin  # n_samples x x_grid
    print("Initial condition shape (n_samples x X_grid): ", init_conds.shape)
    params_a = samples[:, 1].view(-1, 1)
    params_a = (
        params_a.reshape(-1, 1, 1, params_a.shape[-1])
        .repeat(1, n_stoch_samples, 1, 1)
        .repeat(1, 1, n_grids, 1)
    )
    params_a = params_a.view(
        -1, n_grids, params_a.shape[-1]
    )  # n_sample+stoch_samples x n_grids x d
    print(f"params_a shape: {params_a.shape}")

    # Complete model simulations with sampled convection term (stochastic one-to-many)
    unknown_params_sampler = BoxUniformSampler(
        c_coeff_lb, c_coeff_ub, device=device
    )

    # Sample unknown parameters
    # single sample or not
    if fixed_stoch_samples:
        unknown_params = unknown_params_sampler.sample(
            (n_stoch_samples,)
        ).repeat(n_samples, 1)
        print(unknown_params.shape)
    else:
        unknown_params = unknown_params_sampler.sample(
            (n_samples, n_stoch_samples)
        )
    unknown_params = unknown_params.reshape(
        -1, 1, unknown_params.shape[-1]
    ).repeat(1, n_grids, 1)

    print(f"unknown_params shape: {unknown_params.shape}")

    # Broadcast init_conds for noisy samples
    new_init_cond = init_conds.reshape(-1, 1, init_conds.shape[-1])
    new_init_cond = new_init_cond.repeat(1, n_stoch_samples, 1)
    new_init_cond = new_init_cond.view(-1, new_init_cond.shape[-1])
    print(f"Final new_init_cond shape: {new_init_cond.shape}")

    # Full parameters (known + unknown)
    params = torch.cat([params_a, unknown_params], dim=-1)
    print(f"Final params shape: {params.shape}")

    # Check for the shape and consistency of params_a in params
    assert torch.isclose(
        params[:, :, :1], params_a.reshape(-1, n_grids, params_a.shape[-1])
    ).all()
    assert (
        params_a.reshape(-1, n_stoch_samples, n_grids, params_a.shape[-1])[0]
        .unique(dim=0)
        .shape[1]
        == n_grids
    )
    assert (
        params_a.reshape(-1, n_stoch_samples, n_grids, params_a.shape[-1])[
            :, :, 0, :
        ]
        .unique(dim=0)
        .shape[1]
        == n_stoch_samples
    )
    del params_a, unknown_params

    # Simulate by batches
    solver = AdvDiffSolver(
        dx=dx,
        dt=dt,
        len_episode=len_episode,
        n_grids=n_grids,
        noise_std=noise_std,
        device=device,
        method=method,
    )

    simulations = torch.nan * torch.ones(
        (new_init_cond.shape[0], n_grids, len_episode)
    )
    func_eval_time_hist = torch.zeros(
        ceil(new_init_cond.shape[0] // batch_size)
    )

    for i in range(0, new_init_cond.shape[0], batch_size):
        print(
            f"Batch Progress {i} / {params.shape[0]} = {i/params.shape[0]:.2f}"
        )
        print(
            new_init_cond[i : i + batch_size].shape,
            params[i : i + batch_size].shape,
        )
        y_res = solver(
            new_init_cond[i : i + batch_size],
            params[i : i + batch_size].view(-1, params.shape[-1]),
        )
        simulations[i : i + batch_size] = y_res["x"].clone()
        func_eval_time_hist[i : i + batch_size] = y_res["time_eval"]

    assert torch.isfinite(simulations).all()
    assert simulations.shape[0] == new_init_cond.shape[0]
    assert simulations.shape[1] == n_grids
    assert simulations.shape[2] == len_episode

    dic_data = {
        "x": simulations,
        "init_conds": new_init_cond,
        "params": params,
        "x_grid": x_grid,
        "t": solver.t,
    }

    return dic_data


if __name__ == "__main__":
    parser = AdvDiffOptionParser()
    args = parser.parse_args()

    # set random seed
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

    # generate data
    kwargs = vars(args)
    # Print with nice format the arguments
    print("Arguments: ")
    for k, v in kwargs.items():
        print(f"\t {k}: {v}")

    res = generate_data(**kwargs)

    # Hashing the experiment configuration
    min_conf = AdvDiffOptionParser.get_min_dict(args)

    # compute the hash of the experiment and use it as the file name
    hashed_exp_name = hash.uuid_hash(min_conf)
    print(f"Hashed experiment name: {hashed_exp_name}")
    args.filename = f"{hashed_exp_name}"

    args_exp = {
        "all_args": vars(args),
        HASHED_KEYS_NAME: min_conf,
    }

    os.makedirs(args.outdir, exist_ok=True)
    # save args
    with open(
        "{}/args_{}.json".format(args.outdir, hashed_exp_name), "w"
    ) as f:
        json.dump(args_exp, f, indent=4)

    # save data
    torch.save(res, "{}/data_{}".format(args.outdir, hashed_exp_name))
    print(f"Simulations results saved in {args.outdir}/data_{hashed_exp_name}")
