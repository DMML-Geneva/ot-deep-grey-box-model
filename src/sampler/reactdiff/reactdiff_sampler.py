from src.sampler.simulator_sampler import SimulationSampler

import torch


class ReactDiffSampler(SimulationSampler):

    def __init__(
        self,
        init_cond_sampler,
        params_sampler,
        sim_solver,
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

    def sample(self, size=-1):
        assert size <= self.batch_size
        n_samples = size if size > 0 else self.batch_size

        # Sample initial conditions
        init_conds = self.init_cond_sampler.sample((n_samples,))
        init_conds = init_conds.view(n_samples, 2, 32, 32)
        # Sample parameters
        params = self.params_sampler.sample((n_samples,))

        # Solve the forward model for dt time steps
        if self.sim_solver is not None:
            result = self.sim_solver(init_conds=init_conds, params=params)
            return {
                "init_conds": init_conds,
                "params": params,
                "x": result["x"],
                "t": result["t"],
                "solv_sol": result["sol"],
            }
        else:
            return {
                "init_conds": init_conds,
                "params": params,
                "x": None,
                "t": None,
                "solv_sol": None,
            }

    def __len__(self):
        return self.n_val_batches

    def reset(
        self,
    ):
        pass
