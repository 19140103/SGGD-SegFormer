# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.model import is_model_wrapper


@HOOKS.register_module()
class FreezeBackboneHook(Hook):
    """Freeze ``model.backbone`` before training starts.

    This keeps pretrained encoders fixed while training lightweight decoder-side
    modules, which is useful for small datasets and low-memory experiments.
    """

    priority = 'VERY_HIGH'

    def before_train(self, runner) -> None:
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        backbone = getattr(model, 'backbone', None)
        if backbone is None:
            runner.logger.warning('FreezeBackboneHook: model has no backbone.')
            return
        backbone.eval()
        frozen = 0
        for param in backbone.parameters():
            param.requires_grad = False
            frozen += param.numel()
        runner.logger.info(
            f'FreezeBackboneHook: froze {frozen:,} backbone parameters.')

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch=None) -> None:
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        backbone = getattr(model, 'backbone', None)
        if backbone is not None:
            backbone.eval()
