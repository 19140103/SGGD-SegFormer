# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


def _binary_probs_and_target(pred, target, ignore_index):
    """Prepare sigmoid probabilities and binary targets for one-channel heads."""
    if pred.dim() == 4 and pred.size(1) == 1:
        pred = pred.squeeze(1)
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)

    valid = (target != ignore_index).float()
    target = torch.where(target == ignore_index, torch.zeros_like(target),
                         target).float()
    probs = pred.sigmoid()
    return probs, target, valid


@MODELS.register_module()
class BinaryDiceLoss(nn.Module):
    """Dice loss for one-channel binary segmentation logits."""

    def __init__(self,
                 smooth=1.0,
                 eps=1e-6,
                 loss_weight=1.0,
                 ignore_index=255,
                 loss_name='loss_binary_dice'):
        super().__init__()
        self.smooth = smooth
        self.eps = eps
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None,
                **kwargs):
        ignore_index = self.ignore_index if ignore_index is None else ignore_index
        probs, target, valid = _binary_probs_and_target(pred, target,
                                                        ignore_index)
        probs = probs * valid
        target = target * valid
        dims = tuple(range(1, probs.dim()))
        inter = (probs * target).sum(dim=dims)
        denom = probs.sum(dim=dims) + target.sum(dim=dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth + self.eps)
        return self.loss_weight * (1 - dice).mean()

    @property
    def loss_name(self):
        return self._loss_name


@MODELS.register_module()
class BinaryTverskyLoss(nn.Module):
    """Tversky loss for crack recall-oriented binary segmentation."""

    def __init__(self,
                 alpha=0.3,
                 beta=0.7,
                 smooth=1.0,
                 eps=1e-6,
                 loss_weight=1.0,
                 ignore_index=255,
                 loss_name='loss_binary_tversky'):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.eps = eps
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None,
                **kwargs):
        ignore_index = self.ignore_index if ignore_index is None else ignore_index
        probs, target, valid = _binary_probs_and_target(pred, target,
                                                        ignore_index)
        probs = probs * valid
        target = target * valid
        dims = tuple(range(1, probs.dim()))
        tp = (probs * target).sum(dim=dims)
        fp = (probs * (1 - target) * valid).sum(dim=dims)
        fn = ((1 - probs) * target).sum(dim=dims)
        score = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth + self.eps)
        return self.loss_weight * (1 - score).mean()

    @property
    def loss_name(self):
        return self._loss_name


def _soft_erode(img):
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.minimum(p1, p2)


def _soft_dilate(img):
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def _soft_open(img):
    return _soft_dilate(_soft_erode(img))


def _soft_skeletonize(img, iters):
    opened = _soft_open(img)
    skeleton = F.relu(img - opened)
    for _ in range(iters):
        img = _soft_erode(img)
        opened = _soft_open(img)
        delta = F.relu(img - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


@MODELS.register_module()
class SoftClDiceLoss(nn.Module):
    """Soft clDice loss for thin-structure topological continuity."""

    def __init__(self,
                 iter_=10,
                 smooth=1.0,
                 eps=1e-6,
                 loss_weight=1.0,
                 ignore_index=255,
                 loss_name='loss_cldice'):
        super().__init__()
        self.iter_ = iter_
        self.smooth = smooth
        self.eps = eps
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None,
                **kwargs):
        ignore_index = self.ignore_index if ignore_index is None else ignore_index
        probs, target, valid = _binary_probs_and_target(pred, target,
                                                        ignore_index)
        probs = (probs * valid).unsqueeze(1)
        target = (target * valid).unsqueeze(1)
        skel_pred = _soft_skeletonize(probs, self.iter_)
        skel_true = _soft_skeletonize(target, self.iter_)

        dims = tuple(range(1, probs.dim()))
        tprec = ((skel_pred * target).sum(dim=dims) + self.smooth) / (
            skel_pred.sum(dim=dims) + self.smooth + self.eps)
        tsens = ((skel_true * probs).sum(dim=dims) + self.smooth) / (
            skel_true.sum(dim=dims) + self.smooth + self.eps)
        cldice = (2 * tprec * tsens) / (tprec + tsens + self.eps)
        return self.loss_weight * (1 - cldice).mean()

    @property
    def loss_name(self):
        return self._loss_name
