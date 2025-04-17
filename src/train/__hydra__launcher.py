import os
from os.path import dirname

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from src.io.utils import get_hydra_output_dir

from src.train.runner.algorithm_runner import run_algorithm
from src.utils import hash

config_dir = os.path.join(dirname(dirname(hash.__file__)), "conf")


@hydra.main(version_base="1.3", config_path=config_dir, config_name="default")
def main(cfg: DictConfig) -> None:
    """
    Run benchmarking.
    """
    print(OmegaConf.to_yaml(cfg))
    if cfg.outdir is None or cfg.outdir == "" or cfg.outdir.lower() == "none":
        with open_dict(cfg):
            cfg.outdir = get_hydra_output_dir()

    run_algorithm(cfg)


if __name__ == "__main__":
    main()
