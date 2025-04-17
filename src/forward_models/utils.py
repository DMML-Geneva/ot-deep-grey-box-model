import contextlib
import multiprocessing as mp
import operator
import os
import random
from collections import OrderedDict
from collections.abc import Callable
from functools import reduce

import numpy
import numpy as np
import torch
import wandb


@contextlib.contextmanager
def modified_environ(*remove, **update):
    """
    Copied from: https://stackoverflow.com/a/34333710

    Temporarily updates the ``os.environ`` dictionary in-place.

    The ``os.environ`` dictionary is updated in-place so that the modification
    is sure to work in all situations.

    :param remove: Environment variables to remove.
    :param update: Dictionary of environment variables and values to add/update.
    """
    env = os.environ
    update = update or {}
    remove = remove or []

    # List of environment variables being updated or removed.
    stomped = (set(update.keys()) | set(remove)) & set(env.keys())
    # Environment variables and values to restore on exit.
    update_after = {k: env[k] for k in stomped}
    # Environment variables and values to remove on exit.
    remove_after = frozenset(k for k in update if k not in env)

    try:
        env.update(update)
        [env.pop(k, None) for k in remove]
        yield
    finally:
        env.update(update_after)
        [env.pop(k) for k in remove_after]


def seed_dataloader_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


class DefaultOrderedDict(OrderedDict):
    # Source: http://stackoverflow.com/a/6190500/562769
    def __init__(self, default_factory=None, *a, **kw):
        if default_factory is not None and not isinstance(
            default_factory, Callable
        ):
            raise TypeError("first argument must be callable")
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = (self.default_factory,)
        return type(self), args, None, None, iter(self.items())

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        import copy

        return type(self)(self.default_factory, copy.deepcopy(self.items()))

    def __repr__(self):
        return "OrderedDefaultDict(%s, %s)" % (
            self.default_factory,
            OrderedDict.__repr__(self),
        )


def set_seed(seed):
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        env_context = modified_environ(CUBLAS_WORKSPACE_CONFIG=":4096:8")
    else:
        env_context = contextlib.nullcontext
    return env_context


def split_path(path):
    # source: https://www.oreilly.com/library/view/python-cookbook/0596001673/ch04s16.html
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def normalize_dict(input):
    if input.pop("normalize", True):
        return {k: v / sum(input.values()) for k, v in input.items()}
    else:
        return input


def linspace(start, stop, N, endpoint=True):
    """Returns linspace for multiple inputs."""
    # source: https://stackoverflow.com/questions/40624409/vectorized-numpy-linspace-for-multiple-start-and-stop-values
    if endpoint == 1:
        divisor = N - 1
    else:
        divisor = N
    steps = (1.0 / divisor) * (stop - start)
    return (
        steps[:, None] * torch.arange(N, device=start.device) + start[:, None]
    )


def set_precision(precision):
    if precision == 16:
        torch.set_default_dtype(torch.float16)
    elif precision == 32:
        torch.set_default_dtype(torch.float32)
    elif precision == 64:
        torch.set_default_dtype(torch.float64)
    else:
        raise ValueError(f"Unknown precision value: {precision}")


def map_mean_shift(
    samples,
    y_gt=None,
    kernel=lambda x: torch.exp(-1.0 * (x**2).sum(-1)),
    max_iter=500,
):
    eps = 1e-6
    m = samples[torch.randint(0, high=samples.shape[0], size=(1,)).item(), :]
    m_old = m - 1000
    i = 0
    while ((m - m_old) ** 2).sum().sqrt() > eps and i < max_iter:
        neighbourhood = samples[~(m_old == samples).all(-1)]
        ker_val = kernel(neighbourhood - m)
        m_new = (ker_val.unsqueeze(-1) * neighbourhood).sum(0) / ker_val.sum()
        m_old = m
        m = m_new
        i += 1
    return m


def map_flow(samples, y_gt, log_density_fn, model, model_dims, device, c=None):
    samples_density = log_density_fn(
        samples,
        y_gt.unsqueeze(0)
        .expand(samples.shape[0], *[-1] * y_gt.ndim)
        .to(device),
        model,
        model_dims,
        device,
        c=c,
    )
    return samples[samples_density.argmax().item()]


def report_posterior_comparison_metrics_to_wandb(posterior_comparison_metrics):
    for family, stats in posterior_comparison_metrics.items():
        for stat, val in stats.items():
            wandb.run.summary[f"test/posterior_comparison/{family}/{stat}"] = (
                val
            )


def report_posterior_metrics_to_wandb(posterior_metrics):
    for param_family, types_crs in posterior_metrics[
        "calibration_error"
    ].items():
        for type_name, crs in types_crs.items():
            wandb.run.summary[
                f"test/posterior/calibration_error/{param_family}/{type_name}/median"
            ] = crs["median"]
            wandb.run.summary[
                f"test/posterior/calibration_error/{param_family}/{type_name}/median-relu"
            ] = crs["median-relu"]
    for map_type, idxs in posterior_metrics["rmse"]["x"].items():
        for idx, vals in idxs.items():
            for val_type, val in vals.items():
                wandb.run.summary[
                    f"test/posterior/rmse/x/{idx}/{val_type}"
                ] = val
    for idx, vals in posterior_metrics["rmse"]["y"][
        "forward-mean-shift"
    ].items():
        for val_type, val in vals.items():
            wandb.run.summary[
                f"test/posterior/rmse/y/forward-mean-shift/{idx}/{val_type}"
            ] = val
    for map_type, idxs in posterior_metrics["rmse"]["y"]["x_map"].items():
        for idx, vals in idxs.items():
            for val_type, val in vals.items():
                wandb.run.summary[
                    f"test/posterior/rmse/y/x_map/{map_type}/{idx}/{val_type}"
                ] = val


def get_from_dict(data_dict, keys_list):
    return reduce(operator.getitem, keys_list, data_dict)


class DifferentiableClamp(torch.autograd.Function):
    """
    Source: https://discuss.pytorch.org/t/exluding-torch-clamp-from-backpropagation-as-tf-stop-gradient-in-tensorflow/52404/6
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    """

    @staticmethod
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def dclamp(input, min=None, max=None):
    """
    Like torch.clamp, but with a constant 1-gradient.
    :param input: The input that is to be clamped.
    :param min: The minimum value of the output.
    :param max: The maximum value of the output.
    """
    return DifferentiableClamp.apply(input, min, max)


def get_num_avialable_cpus():
    """Calculates number of CPUs in a robust way."""
    # Based on pytorch DataLoader class implementation
    max_num_worker_suggest = None
    if hasattr(os, "sched_getaffinity"):
        try:
            max_num_worker_suggest = len(os.sched_getaffinity(0))
        except Exception:
            pass
    if max_num_worker_suggest is None:
        # os.cpu_count() could return Optional[int]
        # get cpu count first and check None in order to satify mypy check
        cpu_count = os.cpu_count()
        if cpu_count is not None:
            max_num_worker_suggest = cpu_count
        else:
            max_num_worker_suggest = mp.cpu_count()
    return max_num_worker_suggest


def get_unique_last_idx(initial_list):
    a = np.asarray(initial_list)
    mask = np.concatenate(([True], a[1:] != a[:-1], [True]))
    idx = np.flatnonzero(mask)
    l = np.diff(idx)
    ss = np.repeat(idx[1:] - 1, l, axis=0)
    return np.unique(ss)


def wandb_log_params(models, log="all"):
    if not isinstance(models, (tuple, list)):
        models = (models,)
    for local_idx, model in enumerate(models):
        # Prefix used for name consistency with wandb convention.
        # Copied from wandb_watch.py:L82-L87
        if local_idx > 0:
            # This makes ugly chart names like gradients/graph_1conv1d.bias
            prefix = "graph_%i" % local_idx
        else:
            prefix = ""
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                if log in ("parameters", "all"):
                    wandb.run._torch.log_tensor_stats(
                        parameter, "parameters/" + prefix + name
                    )
                if log in ("gradients", "all") and parameter.grad is not None:
                    wandb.run._torch.log_tensor_stats(
                        parameter.grad.data, "gradients/" + prefix + name
                    )


class ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scaling=1.0):
        ctx.scaling = scaling
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scaling, None


def setup_torch():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


EPS = 1e-6

import time


class Timer:
    """
    A small Timer class used to time different possible expensive functio.
    """

    def __init__(self, eps_t=1e-9):
        """
        Initialize a new timer.
        """
        self._start_times = dict()
        self._durations = dict()
        self.eps_t = eps_t

    def start_timer(self, name: str):
        """
        Start the specified timer.

        Parameters
        ----------
        name : str
            The name of the timer that should be started.
        """
        if name not in self._start_times:
            if name in self._durations:
                self._durations.pop(name)
            self._start_times[name] = time.perf_counter()

    def stop_timer(self, name: str):
        """
        Stop the specified timer

        Parameters
        ----------
        name : str
            The name of the timer that should be started.
        """

        if name in self._start_times:
            end_time = time.perf_counter()
            self._durations[name] = (
                end_time - self._start_times[name]
            ) + self.eps_t
            self._start_times.pop(name)

    def get_duration(self, name: str):
        """
        Return the duration of the specified timer.

        Parameters
        ----------
        name : str
            The name of the timer which time should be returned.

        Returns
        -------
        duration : float
            The duration of the timer or None when the timer is not existing.
        """
        return self._durations.get(name)
