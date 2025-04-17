from src.forward_models.pendulum import PendulumSolver, SimplePendulum
from src.forward_models.advdiff import AdvDiffSolver
from src.forward_models.reactdiff import ReactDiffSolver


def init_physics_solver_model(config, device):
    # Initialize the physics model
    if config["data_sampler"]["name"] == "pendulum":
        phys_solver = PendulumSolver(
            len_episode=config["data_sampler"]["forward_model"]["f_evals"],
            dt=config["data_sampler"]["forward_model"]["dt"],
            method=(
                config["data_sampler"]["forward_model"]["method"]
                if config["data_sampler"]["online"]
                else "euler"
            ),
        ).to(device)
    elif config["data_sampler"]["name"] == "advdiff":
        phys_solver = AdvDiffSolver(
            dx=config["data_sampler"]["forward_model"]["dx"],
            dt=config["data_sampler"]["forward_model"]["dt"],
            n_grids=config["data_sampler"]["forward_model"]["n_grids"],
            len_episode=config["data_sampler"]["forward_model"]["f_evals"],
            method=(
                config["data_sampler"]["forward_model"]["method"]
                if config["data_sampler"]["online"]
                else "euler"
            ),
        ).to(device)
    elif config["data_sampler"]["name"] == "reactdiff":
        cfg_forward_model = config["data_sampler"]["forward_model"]
        phys_solver = ReactDiffSolver(
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
    else:
        phys_solver = None

    return phys_solver
