# Copyright (c) OpenMMLab. All rights reserved.
"""Structure-guided gated decoder for bridge crack segmentation.

The implementation keeps the original SPG-SegFormer checkpoint interface, but
the paper-facing method is a complete Structure-Guided Gated Decoder (SGGD):
direction-aware structure prior estimation, multi-scale prior-guided fusion,
and gated bottleneck refinement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize


def _direction_specs(main_kernel_size=7, cross_kernel_size=3):
    assert main_kernel_size % 2 == 1
    assert cross_kernel_size % 2 == 1
    main_pad = main_kernel_size // 2
    cross_pad = cross_kernel_size // 2
    return [
        ((1, main_kernel_size), (0, main_pad)),
        ((main_kernel_size, 1), (main_pad, 0)),
        ((cross_kernel_size, main_kernel_size), (cross_pad, main_pad)),
        ((main_kernel_size, cross_kernel_size), (main_pad, cross_pad)),
    ]


def _as_tuple(value):
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value, )


class DirectionAwareStructurePrior(nn.Module):
    """Generate a crack structure prior from high-resolution decoder features."""

    def __init__(self,
                 in_channels,
                 prior_channels=64,
                 direction_kernel_size=7,
                 direction_kernel_sizes=None,
                 direction_cross_size=3,
                 dynamic_branch_selection=False,
                 branch_gate_mode='softmax',
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        direction_kernel_sizes = _as_tuple(
            direction_kernel_size
            if direction_kernel_sizes is None else direction_kernel_sizes)
        self.reduce = ConvModule(
            in_channels,
            prior_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.direction_convs = nn.ModuleList()
        for main_kernel_size in direction_kernel_sizes:
            for kernel_size, padding in _direction_specs(
                    main_kernel_size, direction_cross_size):
                self.direction_convs.append(
                    ConvModule(
                        prior_channels,
                        prior_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        groups=prior_channels,
                        norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        self.dynamic_branch_selection = dynamic_branch_selection
        self.branch_gate_mode = branch_gate_mode
        if self.dynamic_branch_selection:
            self.branch_gate = nn.Conv2d(
                prior_channels, len(self.direction_convs), kernel_size=1)
        else:
            self.branch_gate = None
        self.mix = ConvModule(
            prior_channels * len(self.direction_convs),
            prior_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.prior_pred = nn.Conv2d(prior_channels, 1, kernel_size=1)

    def forward(self, x):
        x = self.reduce(x)
        directional_feats = [conv(x) for conv in self.direction_convs]
        if self.branch_gate is not None:
            branch_logits = self.branch_gate(x)
            if self.branch_gate_mode == 'softmax':
                branch_weights = torch.softmax(branch_logits, dim=1)
            elif self.branch_gate_mode == 'sigmoid':
                branch_weights = torch.sigmoid(branch_logits)
            elif self.branch_gate_mode == 'sigmoid_residual':
                branch_weights = 1.0 + torch.sigmoid(branch_logits)
            else:
                raise ValueError(f'Unsupported branch_gate_mode: '
                                 f'{self.branch_gate_mode}')
            directional_feats = [
                feat * branch_weights[:, idx:idx + 1]
                for idx, feat in enumerate(directional_feats)
            ]
        prior_feat = self.mix(torch.cat(directional_feats, dim=1))
        prior_logits = self.prior_pred(prior_feat)
        prior = torch.sigmoid(prior_logits)
        return prior, prior_feat, prior_logits


class GatedBottleneckRefinement(nn.Module):
    """Lightweight gated refinement for high-resolution fused features."""

    def __init__(self,
                 channels,
                 reduction=4,
                 min_channels=32,
                 direction_kernel_size=7,
                 direction_cross_size=3,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        hidden_channels = max(channels // reduction, min_channels)
        self.reduce = ConvModule(
            channels,
            hidden_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.direction_convs = nn.ModuleList()
        for kernel_size, padding in _direction_specs(direction_kernel_size,
                                                    direction_cross_size):
            self.direction_convs.append(
                ConvModule(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    groups=hidden_channels,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        self.mix = ConvModule(
            hidden_channels * len(self.direction_convs),
            hidden_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.refine = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.feature_gate = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.prior_gate = nn.Conv2d(1, channels, kernel_size=1)
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, prior):
        bottleneck = self.reduce(x)
        directional_feats = [conv(bottleneck) for conv in self.direction_convs]
        bottleneck = self.mix(torch.cat(directional_feats, dim=1))
        refine = self.refine(bottleneck)
        gate = torch.sigmoid(self.feature_gate(bottleneck) +
                             self.prior_gate(prior))
        return x + self.res_scale * gate * refine


@MODELS.register_module()
class SPGSegformerHead(BaseDecodeHead):
    """Structure-guided gated SegFormer decoder head.

    The head keeps SegFormer's MLP projection/fusion style, but organizes the
    crack-specific prior and refinement as one decoder:

    1. direction-aware structure prior estimation;
    2. prior-guided multi-scale feature modulation;
    3. gated bottleneck local refinement.
    """

    def __init__(self,
                 interpolate_mode='bilinear',
                 prior_channels=64,
                 prior_strength=0.5,
                 prior_suppression_strength=0.0,
                 enable_prior=True,
                 enable_refinement=True,
                 refinement_reduction=4,
                 direction_kernel_size=7,
                 prior_direction_kernel_sizes=None,
                 direction_cross_size=3,
                 dynamic_prior_branch_selection=False,
                 dynamic_prior_gate_mode='softmax',
                 enable_prior_aux_loss=False,
                 prior_aux_loss_weight=0.2,
                 prior_aux_pos_weight=20.0,
                 prior_aux_dice_weight=1.0,
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        self.prior_strength = prior_strength
        self.prior_suppression_strength = prior_suppression_strength
        self.enable_prior = enable_prior
        self.enable_refinement = enable_refinement
        self.enable_prior_aux_loss = enable_prior_aux_loss
        self.prior_aux_loss_weight = prior_aux_loss_weight
        self.prior_aux_pos_weight = prior_aux_pos_weight
        self.prior_aux_dice_weight = prior_aux_dice_weight
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        if self.enable_prior:
            self.structure_prior = DirectionAwareStructurePrior(
                self.channels,
                prior_channels=prior_channels,
                direction_kernel_size=direction_kernel_size,
                direction_kernel_sizes=prior_direction_kernel_sizes,
                direction_cross_size=direction_cross_size,
                dynamic_branch_selection=dynamic_prior_branch_selection,
                branch_gate_mode=dynamic_prior_gate_mode,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            self.prior_gammas = nn.Parameter(
                torch.full((num_inputs, ), float(prior_strength)))
        else:
            self.structure_prior = None
            self.prior_gammas = None

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        if self.enable_refinement:
            self.refinement = GatedBottleneckRefinement(
                self.channels,
                reduction=refinement_reduction,
                direction_kernel_size=direction_kernel_size,
                direction_cross_size=direction_cross_size,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            self.refinement = None

    def forward(self, inputs):
        seg_logits, _ = self._forward_feature(inputs, return_prior_logits=False)
        return seg_logits

    def _forward_feature(self, inputs, return_prior_logits=False):
        # Receive 4 stage backbone feature maps: 1/4, 1/8, 1/16, 1/32.
        inputs = self._transform_inputs(inputs)
        projected = [conv(x) for conv, x in zip(self.convs, inputs)]

        prior = None
        prior_logits = None
        if self.enable_prior:
            prior, _, prior_logits = self.structure_prior(projected[0])

        outs = []
        for idx, x in enumerate(projected):
            if prior is not None:
                prior_for_scale = resize(
                    input=prior,
                    size=x.shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners)
                scale = 1 + self.prior_gammas[idx] * prior_for_scale
                if self.prior_suppression_strength > 0:
                    scale = scale - self.prior_suppression_strength * (
                        1 - prior_for_scale)
                    scale = torch.clamp(scale, min=0.0)
                x = x * scale
            outs.append(
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))

        if self.refinement is not None:
            if prior is None:
                prior = out.new_zeros(
                    (out.shape[0], 1, out.shape[2], out.shape[3]))
            elif prior.shape[2:] != out.shape[2:]:
                prior = resize(
                    input=prior,
                    size=out.shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners)
            out = self.refinement(out, prior)

        out = self.cls_seg(out)
        if return_prior_logits:
            return out, prior_logits
        return out, None

    def loss(self, inputs, batch_data_samples, train_cfg):
        seg_logits, prior_logits = self._forward_feature(
            inputs, return_prior_logits=True)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)

        if self.enable_prior_aux_loss and prior_logits is not None:
            prior_losses = self.loss_prior_by_feat(prior_logits,
                                                   batch_data_samples)
            losses.update(prior_losses)
        return losses

    def loss_prior_by_feat(self, prior_logits, batch_data_samples):
        seg_label = self._stack_batch_gt(batch_data_samples).float()
        prior_logits = resize(
            input=prior_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        valid = seg_label != float(self.ignore_index)
        target = seg_label.clamp(min=0, max=1)
        pos_weight = prior_logits.new_tensor(self.prior_aux_pos_weight)
        loss_bce = F.binary_cross_entropy_with_logits(
            prior_logits,
            target,
            pos_weight=pos_weight,
            reduction='none')
        loss_bce = (loss_bce * valid).sum() / valid.sum().clamp_min(1)

        probs = torch.sigmoid(prior_logits) * valid
        target = target * valid
        dims = tuple(range(1, probs.dim()))
        inter = (probs * target).sum(dim=dims)
        denom = probs.sum(dim=dims) + target.sum(dim=dims)
        loss_dice = 1 - ((2 * inter + 1.0) / (denom + 1.0 + 1e-6)).mean()

        return dict(
            loss_prior_bce=self.prior_aux_loss_weight * loss_bce,
            loss_prior_dice=self.prior_aux_loss_weight *
            self.prior_aux_dice_weight * loss_dice)


@MODELS.register_module()
class StructureGuidedGatedDecoderHead(SPGSegformerHead):
    """Paper-facing alias for the complete SGGD decoder.

    This class intentionally reuses ``SPGSegformerHead`` without changing any
    parameter names, so existing SPG checkpoints remain load-compatible while
    new configs can describe the method as a unified decoder rather than loose
    prior/refinement plug-ins.
    """

    pass
