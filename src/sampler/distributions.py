import torch
from torch.distributions import Uniform


class LoaderSampler:
    def __init__(self, loader, device="cuda"):
        self.device = device
        self.loader = loader
        self.it = iter(self.loader)

    def sample(self, size=-1):
        assert size <= self.loader.batch_size
        try:
            batch = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return self.sample(size)

        if self.loader.batch_size < size:
            return self.sample(size)

        if size > 0 and size < self.loader.batch_size:
            if isinstance(batch, dict):
                for k in batch:
                    batch[k] = batch[k][:size]
            else:
                if size > 0:
                    batch = batch[:size]

        return batch

    def reset(self):
        self.it = iter(self.loader)


class Sampler:
    def __init__(
        self,
        device="cuda",
        seed=None,
    ):
        self.device = torch.device(device)
        self.seed = seed
        self.rnd_generator = torch.Generator(device=device)
        if seed is not None:
            self.rnd_generator.manual_seed(seed)

    def sample(self, size=5):
        pass


class BoxUniformSampler(Sampler):
    def __init__(self, lower_bounds, upper_bounds, seed=None, device="cuda"):
        super().__init__(device=device, seed=seed)
        if isinstance(lower_bounds, float):
            lower_bounds = [lower_bounds]
        if isinstance(upper_bounds, float):
            upper_bounds = [upper_bounds]
        if not isinstance(lower_bounds, torch.Tensor):
            lower_bounds = torch.tensor(lower_bounds, device=device)
        if not isinstance(upper_bounds, torch.Tensor):
            upper_bounds = torch.tensor(upper_bounds, device=device)
        assert lower_bounds.shape == upper_bounds.shape
        assert lower_bounds.ndim == 1
        assert torch.all(lower_bounds <= upper_bounds)

        self.ranges = torch.stack((lower_bounds, upper_bounds), dim=1)
        # Remove constant dimensions
        self.init_ranges = self.ranges[self.ranges[:, 0] != self.ranges[:, 1]]
        self.uniform_sampler = Uniform(
            self.init_ranges[:, 0], self.init_ranges[:, 1]
        )
        self.constant_positions = torch.where(
            self.ranges[:, 0] == self.ranges[:, 1]
        )[0]
        self.non_constant_positions = torch.where(
            self.ranges[:, 0] != self.ranges[:, 1]
        )[0]
        self.device = device

    def sample(self, size):
        samples = self.uniform_sampler.sample(size)

        if samples.ndim > 2:
            samples = samples.reshape(-1, samples.shape[-1])

        if len(self.constant_positions) > 0:
            res = torch.empty(
                samples.shape[0], self.ranges.shape[0], device=self.device
            )
            res[:, self.non_constant_positions] = samples
        else:
            return samples

        res[:, self.constant_positions] = self.ranges[
            self.constant_positions, 0
        ]
        return res.reshape(size + (-1,))


class Bernoulli(Sampler):
    def __init__(
        self,
        p=0.5,
        head=1,
        tail=0,
        seed=None,
        device="cpu",
    ):
        super().__init__(seed=seed, device=device)
        self.head = head
        self.head = torch.atleast_2d(head)
        self.tail = torch.atleast_2d(tail)
        self.p = p
        if isinstance(self.p, list):
            self.p = self.p[0]
        if torch.is_tensor(self.p):
            self.p = self.p.item()

    def sample(self, size=(1,)):
        if isinstance(size, int):
            size = (size, 1)

        tosses = torch.bernoulli(
            self.p * torch.ones(size, device=self.device),
            generator=self.rnd_generator,
        )
        tosses = tosses.view(-1, 1).to(bool)
        samples = self.head * tosses
        samples += self.tail * ~tosses
        return samples.reshape(size + (self.head.shape[-1],))


class Categorical(Sampler):
    def __init__(
        self,
        probs=[0.5],
        classes=[0, 1],
        seed=None,
        device="cpu",
    ):
        super().__init__(seed=seed, device=device)
        self.probs = torch.tensor(probs, device=device)
        assert self.probs.sum() == 1
        self.classes = torch.tensor(classes, device=device)
        assert self.classes.shape[0] == self.probs.shape[0]
        self.categorical = torch.distributions.Categorical(
            probs=torch.tensor(probs, device=device)
        )

    def sample(self, size=(1,)):
        samples = self.categorical.sample(size)
        return self.classes[samples]


class LoaderSampler(Sampler):
    def __init__(self, loader, device="cuda", seed=None):
        super(LoaderSampler, self).__init__(device=device, seed=seed)
        self.loader = loader
        self.it = iter(self.loader)

    def sample(self, size=5):
        assert size <= self.loader.batch_size
        try:
            batch, _ = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return self.sample(size)
        if len(batch) < size:
            return self.sample(size)

        return batch[:size].to(self.device)

class StandardNormalSampler(Sampler):
    def __init__(self, dim=1, device="cuda", seed=None):
        super(StandardNormalSampler, self).__init__(device=device, seed=seed)
        self.dim = dim

    def sample(self, batch_size=10):
        return torch.randn(batch_size, self.dim, device=self.device)


def dist_sampler(name, *args):
    if name == "uniform":
        return BoxUniformSampler(*args)
    elif name == "bernoulli":
        return Bernoulli(*args)
    elif name == "categorical":
        return Categorical(*args)
    else:
        raise ValueError(f"Unknown distribution {name}")


def sample_latent_noise(
    n_samples, z_dim=1, z_size=4, z_std=0.1, type="gauss", device="cuda"
):
    if type.lower() == "gauss":
        Z = torch.randn(n_samples, z_size, z_dim, device=device) * z_std
    elif type.lower() == "uniform":
        Z = torch.rand(n_samples, z_size, z_dim, device=device)
    elif type.lower() == "bernoulli":
        bernoulli = Bernoulli(
            p=0.5,
            head=torch.ones(z_dim, device=device),
            tail=torch.zeros(z_dim, device=device),
            device=device,
        )
        Z = bernoulli.sample(n_samples * z_size).reshape(
            n_samples, z_size, z_dim
        )
    else:
        raise ValueError(f"Unknown noise latent distribution {type}")
    return Z
