from src.sampler.simulator_sampler import SimulationSampler

import torch


class AdvDiffSampler(SimulationSampler):

    def __init__(
        self,
        init_cond_sampler,
        params_sampler,
        sim_solver,
        n_grids=20,
        dx=0.1,
        batch_size=1000,
        n_val_batches=1,
        device="cuda",
    ):
        self.device = device
        self.init_cond_sampler = init_cond_sampler
        self.params_sampler = params_sampler
        self.sim_solver = sim_solver
        self.dt = None
        if self.sim_solver is not None:
            self.dt = sim_solver.dt

        self.batch_size = batch_size
        self.n_val_batches = n_val_batches

        self.n_grids = n_grids
        self.dx = dx

    def sample(self, size=-1):
        assert size <= self.batch_size
        n_samples = size if size > 0 else self.batch_size

        # Sample initial conditions
        init_conds = self.init_cond_sampler.sample((n_samples,))

        x_grid = torch.linspace(
            0.0,
            self.dx * (self.n_grids - 1),
            self.n_grids,
            device=init_conds.device,
        )  # 20

        init_sin = torch.sin(x_grid / x_grid[-1] * torch.pi)
        # broadcast init_conds * init_sin for boundaries conditions
        init_conds = init_conds * init_sin  # n_samples x x_grid

        # Sample parameters
        params = self.params_sampler.sample((n_samples,))
        params = params.view(-1, 1, params.shape[-1]).repeat(
            1, self.n_grids, 1
        )

        # Solve the forward model for dt time steps
        if self.sim_solver is not None:
            result = self.sim_solver(init_conds=init_conds, params=params)
            return {
                "init_conds": init_conds,
                "params": params,
                "x": result["x"],
                "t": result["t"],
                "x_grid": result["x_grid"],
                "solv_sol": result["sol"],
            }
        else:
            return {
                "init_conds": init_conds,
                "params": params,
                "x": None,
                "t": None,
                "x_grid": x_grid,
                "solv_sol": None,
            }

    def __len__(self):
        return self.n_val_batches

    def reset(
        self,
    ):
        pass
