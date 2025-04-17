import torch


# Multidimensional Mean square root error
def rmse(y_true, y_pred):
    if y_true.shape == y_pred.shape:
        res = torch.sqrt(torch.mean((y_true - y_pred) ** 2, dim=1))
        return res.mean(), res.std()
    return None


def nrmse(y_true, y_pred):
    ymax = torch.max(y_true)
    ymin = torch.min(y_true)
    if y_true.shape == y_pred.shape:
        res = torch.sqrt(torch.mean((y_true - y_pred) ** 2, dim=1)) / (
            ymax - ymin
        )
        return res.mean(), res.std()
    return None


# Multidimensional Relative Mean square root error
def relative_rmse(y_true, y_pred):
    if y_true.shape == y_pred.shape:
        res = torch.sqrt(
            torch.mean(
                (
                    (torch.tensor(y_true) - torch.tensor(y_pred))
                    / torch.tensor(y_true)
                )
                ** 2,
                dim=1,
            )
        )
        return res.mean(), res.std()
    return None


def abs_err(y_true, y_pred):
    if y_true.shape == y_pred.shape:
        res = torch.mean(torch.abs(y_true - y_pred), dim=1)
        return res.mean(), res.std()

    return None
