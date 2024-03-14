import torch


def create_mask(
    loss,
    forget_rate: float,
):
    mask = None

    num_samples = len(loss)

    ind_sorted = torch.argsort(loss)
    loss_sorted = loss[ind_sorted]
    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_sorted))

    mask = torch.zeros(num_samples, dtype=bool)
    idx_true = ind_sorted[:num_remember]
    mask[idx_true] = 1

    return idx_true
