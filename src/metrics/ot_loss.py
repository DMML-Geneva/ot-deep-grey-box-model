import torch
from torch import nn
from torch import autograd
import torch.nn.functional as F


def angular_metric(X, Y):
    # Polar geodesic distance
    cos_sim = torch.nn.CosineSimilarity(dim=-1)
    return (
        1 - torch.arccos(cos_sim(X, Y)) / torch.pi
    )  # like Universal Sentence Encoder by Google


def ot_knot_dist(T_XZ, X, gamma, COST, ALPHA, Z_SIZE):
    if COST == "weak_mse" or COST == "mse":
        # Weak quadratic cost (normalized by DIM)
        cost_distance = F.mse_loss(X.reshape_as(T_XZ), T_XZ).mean()
        if Z_SIZE > 1:
            T_var = T_XZ.var(dim=1).mean() if gamma > 0 else torch.tensor(0.0)
            T_loss = cost_distance - gamma * T_var
        else:
            T_var = 0.0
            T_loss = cost_distance
    elif COST == "energy":
        # Energy-based quadratic cost (for distance-induced kernel)
        # Equation 15-16 of KNOT
        if gamma > 0:
            T_var = 0.5 * torch.cdist(T_XZ, T_XZ, p=ALPHA).mean() * Z_SIZE
            if Z_SIZE > 1:
                T_var = T_var / (Z_SIZE - 1)
        else:
            T_var = torch.tensor(0.0)

        cost_distance = (
            (X.reshape_as(T_XZ) - T_XZ).norm(dim=-1, p=ALPHA).mean()
        )
        T_loss = cost_distance - gamma * T_var

    elif COST == "gaussian":
        # Gaussian kernel (normalized by DIM)
        idx = torch.triu_indices(Z_SIZE, Z_SIZE, offset=1)
        T_var = (
            1
            - torch.exp(
                -0.5 * (T_XZ[:, idx[0]] - T_XZ[:, idx[1]]).square().mean(dim=2)
            ).mean()
        )
        cost_distance = (
            1 - torch.exp(-0.5 * (X - T_XZ).square().mean(dim=2)).mean()
        )
        T_loss = cost_distance - gamma * T_var
    elif COST == "laplacian":
        # Laplacian kernel (normalized by DIM)
        idx = torch.triu_indices(Z_SIZE, Z_SIZE, offset=1)
        T_var = (
            1
            - torch.exp(
                -(T_XZ[:, idx[0]] - T_XZ[:, idx[1]])
                .square()
                .mean(dim=2)
                .sqrt()
            ).mean()
        )
        cost_distance = (
            1 - torch.exp(-(X - T_XZ).square().mean(dim=2).sqrt()).mean()
        )
        T_loss = cost_distance - gamma * T_var
    else:
        raise Exception("Invalid cost function")

    return T_loss, cost_distance, T_var


def gradient_optimality(f, T_X, X):
    """Gradient Optimality cost for potential"""
    T_X.requires_grad_(True)
    f_G_x = f(T_X)

    gradients = autograd.grad(
        outputs=f_G_x,
        inputs=T_X,
        grad_outputs=torch.ones(f_G_x.size()).to(T_X),
        create_graph=True,
        retain_graph=True,
    )[0]
    mean_x = X.view_as(gradients).mean(dim=0)
    return (gradients.mean(dim=0) - mean_x).norm("fro")
