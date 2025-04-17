import os

import torch.nn as nn


from src.nnets.utils import save_state_model


def wandb_log(wandb, dict_metrics, step):
    if wandb.run is not None:
        wandb.log(dict_metrics, step=step)


def save_checkpoints(
    T,
    f,
    T_opt,
    f_opt,
    step,
    out_path_chkp=None,
    logger=None,
    ending_name="",
):
    freeze(T)
    freeze(f)

    if out_path_chkp is None:
        out_path_chkp = "./"
    chk_dirame = os.path.dirname(out_path_chkp)
    basename = os.path.basename(out_path_chkp)

    check_point_out = os.path.join(chk_dirame, f"{basename}_{ending_name}.pt")
    save_state_model(T, f, step, T_opt, f_opt, check_point_out)
    logger.info(
        f"Saved final checkpoint in: {chk_dirame}, name {check_point_out}, at step {step}"
    )

    unfreeze(T)
    unfreeze(f)
    return check_point_out


def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()


def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train(True)


def weights_init_D(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.kaiming_normal_(
            m.weight, mode="fan_out", nonlinearity="leaky_relu"
        )
    elif classname.find("BatchNorm") != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")


