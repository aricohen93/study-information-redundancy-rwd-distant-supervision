"""Script extracted from
Code for ICML2020 Paper ["Normalized Loss Functions for Deep Learning with Noisy Labels"](https://arxiv.org/abs/2006.13554)
@inproceedings{ma2020normalized,
  title={Normalized Loss Functions for Deep Learning with Noisy Labels},
  author={Ma, Xingjun and Huang, Hanxun and Wang, Yisen and Romano, Simone and Erfani, Sarah and Bailey, James},
  booktitle={ICML},
  year={2020}
}
"""

from numbers import Number
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from ..registry import registry


@registry.loss("PytorchLoss")
class PytorchLoss(_Loss):
    def __init__(self, name=None, **kwargs):
        """
        Returns a PyTorch existing loss

        Parameters
        ----------
        name : str
            The name of the loss

        Returns
        -------
        Loss
        """

        super().__init__()

        pos_weight = kwargs.get("pos_weight")
        if isinstance(pos_weight, Number):
            kwargs["pos_weight"] = torch.tensor(pos_weight)

        self.loss = getattr(torch.nn, name)(**kwargs)

    def forward(self, pred, labels):
        return self.loss(pred, labels)


@registry.loss("MyCELoss")
class MyCELoss(_Loss):
    def __init__(self, reduction="none", weight: Optional[List[int]] = None, **kwargs):
        super(MyCELoss, self).__init__()
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float)
        self.cross_entropy = torch.nn.CrossEntropyLoss(
            reduction=reduction, weight=weight, **kwargs
        )

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        return ce


@registry.loss("SCELoss")
class SCELoss(_Loss):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = -1 * torch.sum(pred * torch.log(label_one_hot), dim=1)

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss


@registry.loss("ReverseCrossEntropy")
class ReverseCrossEntropy(_Loss):
    def __init__(self, num_classes, scale=1.0):
        super(ReverseCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = -1 * torch.sum(pred * torch.log(label_one_hot), dim=1)
        return self.scale * rce.mean()
        normalizor = 1 / 4 * (self.num_classes - 1)
        rce = -1 * torch.sum(pred * torch.log(label_one_hot), dim=1)
        return self.scale * normalizor * rce.mean()


@registry.loss("NormalizedReverseCrossEntropy")
class NormalizedReverseCrossEntropy(_Loss):
    def __init__(self, num_classes, scale=1.0):
        super(NormalizedReverseCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        normalizor = 1 / 4 * (self.num_classes - 1)
        rce = -1 * torch.sum(pred * torch.log(label_one_hot), dim=1)
        return self.scale * normalizor * rce.mean()


@registry.loss("NormalizedCrossEntropy")
class NormalizedCrossEntropy(_Loss):
    def __init__(self, num_classes, scale=1.0, weight: Optional[List[float]] = None):
        super(NormalizedCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

        if weight is not None:
            self.weight = torch.Tensor(weight)
        else:
            self.weight = weight

    def forward(self, pred, labels):
        if self.weight is not None:
            weight = self.weight.to(pred.device)
            pred = F.log_softmax(pred, dim=1) * weight
        else:
            pred = F.log_softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (-pred.sum(dim=1))
        return self.scale * nce.mean()


@registry.loss("GeneralizedCrossEntropy")
class GeneralizedCrossEntropy(_Loss):
    def __init__(self, num_classes, q=0.7):
        super(GeneralizedCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.q = q

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        gce = (1.0 - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return gce.mean()


@registry.loss("NormalizedGeneralizedCrossEntropy")
class NormalizedGeneralizedCrossEntropy(_Loss):
    def __init__(self, num_classes, scale=1.0, q=0.7):
        super(NormalizedGeneralizedCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.q = q
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        numerators = 1.0 - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)
        denominators = self.num_classes - pred.pow(self.q).sum(dim=1)
        ngce = numerators / denominators
        return self.scale * ngce.mean()


@registry.loss("MeanAbsoluteError")
class MeanAbsoluteError(_Loss):
    def __init__(self, num_classes, scale=1.0):
        super(MeanAbsoluteError, self).__init__()
        self.num_classes = num_classes
        self.scale = scale
        return

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        mae = 1.0 - torch.sum(label_one_hot * pred, dim=1)
        return self.scale * mae.mean()


@registry.loss("NormalizedMeanAbsoluteError")
class NormalizedMeanAbsoluteError(_Loss):
    def __init__(self, num_classes, scale=1.0):
        super(NormalizedMeanAbsoluteError, self).__init__()
        self.num_classes = num_classes
        self.scale = scale
        return

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        normalizor = 1 / (2 * (self.num_classes - 1))
        mae = 1.0 - torch.sum(label_one_hot * pred, dim=1)
        return self.scale * normalizor * mae.mean()


@registry.loss("NCEandRCE")
class NCEandRCE(_Loss):
    def __init__(self, alpha, beta, num_classes, weight: Optional[List[float]] = None):
        super(NCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.nce = NormalizedCrossEntropy(
            scale=alpha, num_classes=num_classes, weight=weight
        )
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.rce(pred, labels)


@registry.loss("NCEandMAE")
class NCEandMAE(_Loss):
    def __init__(self, alpha, beta, num_classes):
        super(NCEandMAE, self).__init__()
        self.num_classes = num_classes
        self.nce = NormalizedCrossEntropy(scale=alpha, num_classes=num_classes)
        self.mae = MeanAbsoluteError(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.mae(pred, labels)


@registry.loss("GCEandMAE")
class GCEandMAE(_Loss):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(GCEandMAE, self).__init__()
        self.num_classes = num_classes
        self.gce = GeneralizedCrossEntropy(num_classes=num_classes, q=q)
        self.mae = MeanAbsoluteError(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.gce(pred, labels) + self.mae(pred, labels)


@registry.loss("GCEandRCE")
class GCEandRCE(_Loss):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(GCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.gce = GeneralizedCrossEntropy(num_classes=num_classes, q=q)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.gce(pred, labels) + self.rce(pred, labels)


@registry.loss("GCEandNCE")
class GCEandNCE(_Loss):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(GCEandNCE, self).__init__()
        self.num_classes = num_classes
        self.gce = GeneralizedCrossEntropy(num_classes=num_classes, q=q)
        self.nce = NormalizedCrossEntropy(num_classes=num_classes)

    def forward(self, pred, labels):
        return self.gce(pred, labels) + self.nce(pred, labels)


@registry.loss("NGCEandNCE")
class NGCEandNCE(_Loss):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(NGCEandNCE, self).__init__()
        self.num_classes = num_classes
        self.ngce = NormalizedGeneralizedCrossEntropy(
            scale=alpha, q=q, num_classes=num_classes
        )
        self.nce = NormalizedCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.ngce(pred, labels) + self.nce(pred, labels)


@registry.loss("NGCEandMAE")
class NGCEandMAE(_Loss):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(NGCEandMAE, self).__init__()
        self.num_classes = num_classes
        self.ngce = NormalizedGeneralizedCrossEntropy(
            scale=alpha, q=q, num_classes=num_classes
        )
        self.mae = MeanAbsoluteError(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.ngce(pred, labels) + self.mae(pred, labels)


@registry.loss("NGCEandRCE")
class NGCEandRCE(_Loss):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(NGCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.ngce = NormalizedGeneralizedCrossEntropy(
            scale=alpha, q=q, num_classes=num_classes
        )
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.ngce(pred, labels) + self.rce(pred, labels)


@registry.loss("MAEandRCE")
class MAEandRCE(_Loss):
    def __init__(self, alpha, beta, num_classes):
        super(MAEandRCE, self).__init__()
        self.num_classes = num_classes
        self.mae = MeanAbsoluteError(scale=alpha, num_classes=num_classes)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.mae(pred, labels) + self.rce(pred, labels)


@registry.loss("NLNL")
class NLNL(_Loss):
    def __init__(self, train_loader, num_classes, ln_neg=1):
        super(NLNL, self).__init__()
        self.num_classes = num_classes
        self.ln_neg = ln_neg
        weight = torch.FloatTensor(num_classes).zero_() + 1.0
        if not hasattr(train_loader.dataset, "targets"):
            weight = [1] * num_classes
            weight = torch.FloatTensor(weight)
        else:
            for i in range(num_classes):
                weight[i] = (
                    torch.from_numpy(np.array(train_loader.dataset.targets)) == i
                ).sum()
            weight = 1 / (weight / weight.max())
        self.weight = weight
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.weight)
        self.criterion_nll = torch.nn.NLLLoss()

    def forward(self, pred, labels):
        labels_neg = (
            labels.unsqueeze(-1).repeat(1, self.ln_neg)
            + torch.LongTensor(len(labels), self.ln_neg).random_(1, self.num_classes)
        ) % self.num_classes
        labels_neg = torch.autograd.Variable(labels_neg)

        assert labels_neg.max() <= self.num_classes - 1
        assert labels_neg.min() >= 0
        assert (labels_neg != labels.unsqueeze(-1).repeat(1, self.ln_neg)).sum() == len(
            labels
        ) * self.ln_neg

        s_neg = torch.log(torch.clamp(1.0 - F.softmax(pred, 1), min=1e-5, max=1.0))
        s_neg *= self.weight[labels].unsqueeze(-1).expand(s_neg.size())
        labels = labels * 0 - 100
        loss = self.criterion(pred, labels) * float((labels >= 0).sum())
        loss_neg = self.criterion_nll(
            s_neg.repeat(self.ln_neg, 1), labels_neg.t().contiguous().view(-1)
        ) * float((labels_neg >= 0).sum())
        loss = (loss + loss_neg) / (
            float((labels >= 0).sum()) + float((labels_neg[:, 0] >= 0).sum())
        )
        return loss


@registry.loss("FocalLoss")
class FocalLoss(_Loss):
    """
    https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    """

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * torch.autograd.Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


@registry.loss("NormalizedFocalLoss")
class NormalizedFocalLoss(_Loss):
    def __init__(
        self, scale=1.0, gamma=0, num_classes=10, alpha=None, size_average=True
    ):
        super(NormalizedFocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, input, target):
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        normalizor = torch.sum(-1 * (1 - logpt.data.exp()) ** self.gamma * logpt, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())
        loss = -1 * (1 - pt) ** self.gamma * logpt
        loss = self.scale * loss / normalizor

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


@registry.loss("NFLandNCE")
class NFLandNCE(_Loss):
    def __init__(self, alpha, beta, num_classes, gamma=0.5):
        super(NFLandNCE, self).__init__()
        self.num_classes = num_classes
        self.nfl = NormalizedFocalLoss(
            scale=alpha, gamma=gamma, num_classes=num_classes
        )
        self.nce = NormalizedCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nfl(pred, labels) + self.nce(pred, labels)


@registry.loss("NFLandMAE")
class NFLandMAE(_Loss):
    def __init__(self, alpha, beta, num_classes, gamma=0.5):
        super(NFLandMAE, self).__init__()
        self.num_classes = num_classes
        self.nfl = NormalizedFocalLoss(
            scale=alpha, gamma=gamma, num_classes=num_classes
        )
        self.mae = MeanAbsoluteError(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nfl(pred, labels) + self.mae(pred, labels)


@registry.loss("NFLandRCE")
class NFLandRCE(_Loss):
    def __init__(self, alpha, beta, num_classes, gamma=0.5):
        super(NFLandRCE, self).__init__()
        self.num_classes = num_classes
        self.nfl = NormalizedFocalLoss(
            scale=alpha, gamma=gamma, num_classes=num_classes
        )
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nfl(pred, labels) + self.rce(pred, labels)


@registry.loss("DMILoss")
class DMILoss(_Loss):
    def __init__(self, num_classes):
        super(DMILoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, output, target):
        outputs = F.softmax(output, dim=1)
        targets = target.reshape(target.size(0), 1).cpu()
        y_onehot = torch.FloatTensor(target.size(0), self.num_classes).zero_()
        y_onehot.scatter_(1, targets, 1)
        y_onehot = y_onehot.transpose(0, 1).cuda()
        mat = y_onehot @ outputs
        return -1.0 * torch.log(torch.abs(torch.det(mat.float())) + 0.001)
