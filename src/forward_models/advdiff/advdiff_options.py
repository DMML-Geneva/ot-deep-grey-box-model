import argparse
from argparse import ArgumentParser
from src.forward_models.advdiff.advdiff_model import HASHED_KEYS


class AdvDiffOptionParser:

    def __init__(self):
        self.parser = ArgumentParser()
        self.parser.add_argument(
            "--outdir",
            type=str,
            default="datasets/forward_models/advdiff/one_to_many/training",
        )

        self.parser.add_argument("--n-samples", type=int, default=1000)
        self.parser.add_argument("--n-stoch-samples", type=int, default=5)
        self.parser.add_argument("--n-grids", type=int, default=20)
        self.parser.add_argument("--len-episode", type=int, default=50)
        self.parser.add_argument("--dx", type=float, default=0.1)  # pi/20
        self.parser.add_argument("--dt", type=float, default=0.02)
        self.parser.add_argument(
            "--range-init-mag", type=float, nargs=2, default=[0.5, 1.5]
        )
        self.parser.add_argument(
            "--range-dcoeff", type=float, nargs=2, default=[1e-2, 1e-1]
        )
        self.parser.add_argument(
            "--range-ccoeff", type=float, nargs=2, default=[1e-2, 1e-1]
        )
        self.parser.add_argument(
            "--fixed_stoch_samples", type=bool, default=False
        )
        self.parser.add_argument("--noise-loc", type=float, default=0.0)
        self.parser.add_argument("--noise-std", type=float, default=0.0)
        self.parser.add_argument("--seed", type=int, default=1234)
        self.parser.add_argument("--method", type=str, default="dopri8")
        self.parser.add_argument(
            "--library", type=str, default="torch"
        )  # torch

    def parse_args(self):
        return self.parser.parse_args()

    def get_min_dict(args):
        if isinstance(args, argparse.Namespace):
            dict_args = args.__dict__
        else:
            dict_args = vars(args)

        min_dict = {key: dict_args[key] for key in HASHED_KEYS}
        return min_dict
