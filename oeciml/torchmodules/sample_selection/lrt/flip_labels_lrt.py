import numpy as np
import torch


def lrt_flip_scheme(pred_softlabels_bar, y_tilde, delta):
    """


    The LRT correction scheme.
    pred_softlabels_bar is the prediction of the network which is compared with noisy label y_tilde.
    If the LR is smaller than the given threshhold delta, we reject LRT and flip y_tilde to prediction of pred_softlabels_bar

    Input
    pred_softlabels_bar: rolling average of output after softlayers for past 10 epochs. Could use other rolling windows.
    y_tilde: noisy labels at current epoch
    delta: LRT threshholding

    Output
    y_tilde : new noisy labels after cleanning
    clean_softlabels : softversion of y_tilde


    << Extracted from "Error-Bounded Correction of Noisy Labels
    Songzhu Zheng, Pengxiang Wu, Aman Goswami, Mayank Goswami, Dimitris Metaxas, Chao Chen
    Paper Link Presented at ICML 2020" >>
    """
    ntrain = pred_softlabels_bar.shape[0]
    num_class = pred_softlabels_bar.shape[1]
    n = 0
    changed_idx = []
    for i in range(ntrain):
        cond_1 = not pred_softlabels_bar[i].argmax() == y_tilde[i]
        cond_2 = (
            pred_softlabels_bar[i].max() / pred_softlabels_bar[i][y_tilde[i]] > delta
        )
        if cond_1 and cond_2:
            y_tilde[i] = pred_softlabels_bar[i].argmax()
            n += 1
            changed_idx.append(i)

    print(f"## {n} data points changed ##")
    eps = 1e-2
    clean_softlabels = torch.ones(ntrain, num_class) * eps / (num_class - 1)
    clean_softlabels.scatter_(
        1, torch.tensor(np.array(y_tilde)).reshape(-1, 1), 1 - eps
    )
    return y_tilde, clean_softlabels, changed_idx
