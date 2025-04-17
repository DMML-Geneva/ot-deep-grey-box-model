from src.utils import hash
from src.forward_models.pendulum.options import PendulumOptionParser
from src.forward_models.advdiff.advdiff_model import (
    HASHED_KEYS as AdvDiff_HASHED_KEYS,
)

from src.forward_models.reactdiff.reactdiff_model import (
    HASHED_KEYS as ReactDiff_HASHED_KEYS,
)
import time


def stopping_criterion(args, x_sims_budget, y_sims_budget, start_time, logger):

    lapsed_time = time.time() - start_time
    early_stop = False

    # check if lapsed time is more than approx args["timeout_early_stop"] hours
    if "timeout_early_stop" in args and args["timeout_early_stop"] > 0:
        early_stop = (
            True if lapsed_time / 3600 >= args["timeout_early_stop"] else False
        )
        if early_stop:
            logger.info(
                f"Early stopping due to timeout at {lapsed_time} hours"
            )

    # check x and y simulation budget
    if (
        "x_max_sims_budg" in args
        and args["x_max_sims_budg"] > 0
        and x_sims_budget >= args["x_max_sims_budg"]
    ):
        early_stop = True
        logger.info(
            f"Early stopping due to x simulation budget at {x_sims_budget}, effective x sims budget: {x_sims_budget}"
        )
    elif (
        "y_max_sims_budg" in args
        and args["y_max_sims_budg"] > 0
        and y_sims_budget >= args["y_max_sims_budg"]
    ):
        early_stop = True
        logger.info(
            f"Early stopping due to y simulation budget at {y_sims_budget}, effective y sims budget: {y_sims_budget}"
        )

    return early_stop


def get_uuid_hash_from_task_exp(conf_exp_x, conf_exp_y, config):

    if config["data_sampler"]["name"] == "pendulum":
        u_x, _ = hash.get_uuid_hash_from_config(
            conf_exp_x, PendulumOptionParser.HASHED_KEYS
        )
        u_y, _ = hash.get_uuid_hash_from_config(
            conf_exp_y, PendulumOptionParser.HASHED_KEYS
        )
    elif config["data_sampler"]["name"] == "advdiff":
        u_x, _ = hash.get_uuid_hash_from_config(
            conf_exp_x, AdvDiff_HASHED_KEYS
        )
        u_y, _ = hash.get_uuid_hash_from_config(
            conf_exp_y, AdvDiff_HASHED_KEYS
        )
    elif config["data_sampler"]["name"] == "reactdiff":
        u_x, _ = hash.get_uuid_hash_from_config(
            conf_exp_x, ReactDiff_HASHED_KEYS
        )
        u_y, _ = hash.get_uuid_hash_from_config(
            conf_exp_y, ReactDiff_HASHED_KEYS
        )
    else:
        raise ValueError("data_sampler name not recognized")

    return u_x, u_y
