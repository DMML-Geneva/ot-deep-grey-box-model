import torch


class SimulationSampler:

    def __init__(
        self,
        init_cond_sampler,
        params_sampler,
        sim_solver,
        batch_size,
        x_params_sampler=None,
        params_z_size=1,
        testing_set=False,
        n_val_batches=1,
        device="cuda",
    ):
        self.device = device
        self.init_cond_sampler = init_cond_sampler
        self.params_sampler = params_sampler
        self.x_params_sampler = x_params_sampler
        self.params_z_size = params_z_size
        self.sim_solver = sim_solver
        self.dt = sim_solver.dt if sim_solver is not None else None

        self.batch_size = batch_size
        self.n_val_batches = n_val_batches
        self.testing_set = testing_set

    def sample(self, size=-1):
        assert size <= self.batch_size
        n_samples = size if size > 0 else self.batch_size

        # Sample initial conditions
        init_conds = self.init_cond_sampler.sample((n_samples,))
        init_cond_dim = tuple(list(init_conds.shape[1:]))

        # Sample parameters
        if self.x_params_sampler is not None:
            x_params = self.x_params_sampler.sample(
                (n_samples,)
            )  # (n_samples, dim)
            params = self.params_sampler.sample(
                (n_samples, self.params_z_size)
            )  # (n_samples, params_z_size, dim)

            x_params = x_params.unsqueeze(1).repeat(1, self.params_z_size, 1)
            if not self.testing_set:
                params = params.reshape(n_samples * self.params_z_size, -1)
                x_params = x_params.reshape(n_samples * self.params_z_size, -1)

            params = torch.cat([x_params, params], dim=-1)
            if self.params_z_size > 1:
                init_conds = init_conds.unsqueeze(1).repeat(
                    1, self.params_z_size, 1
                )
                if not self.testing_set:
                    init_conds = init_conds.reshape(
                        n_samples * self.params_z_size, -1
                    )
        else:
            params = self.params_sampler.sample((n_samples,))

        params_dim = params.shape[-1]
        # Solve the forward model for dt time steps
        if self.sim_solver is None:
            x, t, solv_sol = None, None, None
        else:
            result = self.sim_solver(
                init_conds=init_conds.reshape((-1,) + init_cond_dim),
                params=params.reshape(-1, params_dim),
            )
            x = result["x"]
            t = result["t"]
            solv_sol = result["sol"]

        if self.testing_set and self.sim_solver is not None:
            x_dim = x.shape[1:]
            x = x.reshape((-1, self.params_z_size) + x_dim)

        return {
            "init_conds": init_conds,
            "params": params,
            "x": x,
            "t": t,
            "solv_sol": solv_sol,
        }

    def __len__(self):
        return self.n_val_batches

    def reset(
        self,
    ):
        pass
