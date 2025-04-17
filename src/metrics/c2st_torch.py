import logging
from torch import Tensor
import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, Optional

from src.nnets.convnet import Conv1DNet

from torch.utils.data import DataLoader, TensorDataset
from src.nnets.unet_physics import UNetC2STReactionDiffusion


def train_test_split(data, target, test_size=0.25):
    """
    Split the data into training and testing sets
    """
    indices = np.random.permutation(data.shape[0])
    test_size = int(test_size * data.shape[0])
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    X_train = data[train_indices]
    X_test = data[test_indices]
    y_train = target[train_indices].view(-1, 1)
    y_test = target[test_indices].view(-1, 1)
    return X_train, X_test, y_train, y_test


def c2st(
    X: Tensor,
    Y: Tensor,
    seed: int = 1,
    max_batch_size: int = 50000,
    n_folds: int = 4,
    max_iter: int = 10000,
    early_stopping: bool = False,
    n_iter_no_change: int = 50,
    data_split_ratio: int = 0.25,
    tol: float = 1e-4,
    noise_scale=None,
    z_score=True,
    classifier=None,
    logger=None,
) -> Tensor:
    """
    Return accuracy of classifier trained to distinguish samples from supposedly
    two distributions <X> and <Y>. For details on the method, see [1,2].
    If the returned accuracy is 0.5, <X> and <Y> are considered to be from the
    same generating PDF, i.e. they can not be differentiated.
    If the returned accuracy is around 1., <X> and <Y> are considered to be from
    two different generating PDFs.

    """
    ndim = X.shape[-1]
    device = X.device

    if z_score:
        X_mean = torch.mean(X, dim=0)
        X_std = torch.std(X, dim=0)
        X_new = (X - X_mean) / X_std
        Y_new = (Y - X_mean) / X_std

        # Check if normalizing caused some nans.
        if not torch.isnan(X_new).any() and not torch.isnan(Y_new).any():
            X = X_new
            Y = Y_new

    if noise_scale is not None:
        X += noise_scale * torch.randn(X.shape)
        Y += noise_scale * torch.randn(Y.shape)

    # prepare data
    data = torch.concat((X, Y), dim=0)
    # labels
    target = torch.concat(
        (
            torch.zeros((X.shape[0],), device=device),
            torch.ones((Y.shape[0],), device=device),
        ),
        dim=0,
    )
    # shuffle the data
    indices = torch.randperm(data.shape[0])
    data = data[indices]
    target = target[indices]

    if logger is None:
        logger = logging.getLogger("ADV-KNOT")
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler())
        logger.setLevel("INFO")

    k_fold_scores = []
    for i in range(n_folds):
        # k fold cross validation
        # Split the data for validation if needed
        if data_split_ratio > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                data, target, test_size=data_split_ratio
            )
        else:
            X_train, y_train = data, target.unsqueeze(dim=1)
            X_test, y_test = None, None

        # Define the model
        if classifier is None:
            # classifier = nn.Sequential(
            #   nn.Linear(ndim, ndim // 2),
            #   nn.LeakyReLU(),
            #   nn.Linear(ndim // 2, ndim),
            #   nn.LeakyReLU(),
            #   nn.Linear(ndim, ndim),
            #   nn.LeakyReLU(),
            #   nn.Linear(ndim, 1),
            #   nn.Sigmoid(),
            # )
            classifier = Conv1DNet(
                x_dim=ndim,
                out_dim=1,
                sigmoid=True,
            )
            # classifier = UNetC2STReactionDiffusion(cond_dim=2, t_dim=16)
        classifier = classifier.to(device)
        loss_fn = nn.BCELoss()

        opt = torch.optim.Adam(classifier.parameters(), lr=0.001)

        # Create a data loader for mini-batch training
        sampler = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=int(max_batch_size),
            shuffle=True,
        )
        logger.info(f"c2st: X_train shape: {X_train.shape}")
        logger.info(
            f"c2st: training on k-fold {i} with batch size {int(max_batch_size)}"
        )

        # Train the model
        early_stop_metrics = []
        tr_loss = []
        epoch = 0
        stop_early = False
        while epoch < max_iter and not stop_early:

            batch_loss = []
            batch_acc = []
            for X_batch, Y_batch in sampler:
                opt.zero_grad()
                y_pred = classifier(X_batch)
                loss = loss_fn(y_pred, Y_batch)
                loss.backward()
                opt.step()
                batch_loss.append(loss.item())
                batch_acc.append(
                    torch.mean(((y_pred > 0.5) == Y_batch).float()).item()
                )
            epoch += 1

            tr_loss.append(np.mean(batch_loss))

            # Test the model on the val test
            if X_test is not None:

                y_pred = classifier(X_test)

                # compute the accuracy
                y_hat = y_pred > 0.5
                acc_score = torch.mean((y_hat == y_test).float())

                # compute the loss
                loss = loss_fn(y_pred, y_test)
                if early_stopping:
                    early_stop_metrics.append(acc_score)
                else:
                    # store the training loss
                    early_stop_metrics = tr_loss

                if epoch % 20 == 0:
                    logger.info(f"c2st: val. loss: {loss.item()}")
                    logger.info(
                        f"c2st : val. scores on X (prediction): {acc_score}"
                    )

                idx_label0 = torch.argwhere(y_test == 0.0)[:, 0]
                idx_label1 = torch.argwhere(y_test == 1.0)[:, 0]
                y_hat_0 = y_pred[idx_label0]
                y_hat_1 = y_pred[idx_label1]
                logger.debug(
                    f"c2st : val. scores class 0 (X): {y_hat_0.mean().item()}"
                )
                logger.debug(
                    f"c2st : val. scores class 1 (Y): {y_hat_1.mean().item()}"
                )

                if epoch > n_iter_no_change:
                    v_metric = torch.tensor(early_stop_metrics)
                    # validation score is not improving by at least tol for n_iter_no_change consecutive epochs
                    if torch.all(
                        v_metric[-n_iter_no_change:] - v_metric[-1] < tol
                    ):
                        k_fold_scores.append(acc_score)
                        # stop the training
                        stop_early = True
                        logger.info(
                            f"c2st: early stopping at epoch {epoch}, training loss tol is less than {tol}"
                        )

                if epoch == max_iter - 1:
                    # check if last epoch
                    k_fold_scores.append(acc_score)

        # k fold validation scores
        logger.info(
            f"c2st val: k-fold {i}, accuracy scores: {k_fold_scores[-1]}"
        )

    if len(k_fold_scores) > 0:
        k_fold_scores = torch.tensor(k_fold_scores)
        scores = k_fold_scores.mean()
        logger.info(
            f"c2st: final accuracy scores among k folds: {k_fold_scores.mean()}, std: {k_fold_scores.std()}"
        )

    return scores, classifier, k_fold_scores.std()
