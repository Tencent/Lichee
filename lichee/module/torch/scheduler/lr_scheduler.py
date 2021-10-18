# -*- coding: utf-8 -*-
import math

from torch.optim.lr_scheduler import LambdaLR

from lichee import plugin


@plugin.register_plugin(plugin.PluginType.MODULE_SCHEDULER, 'constant')
class ConstantLRSchedule(LambdaLR):
    """ Constant learning rate schedule.
    """

    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLRSchedule, self).__init__(optimizer, lambda _: 1.0, last_epoch=last_epoch)

    @classmethod
    def build(cls, optimizer, cfg):
        return cls(optimizer)


@plugin.register_plugin(plugin.PluginType.MODULE_SCHEDULER, 'warmup_constant')
class WarmupConstantSchedule(LambdaLR):
    """ Linear warmup and then constant.
        Multiplies the learning rate defined in the optimizer by a dynamic variable determined by the current step.
        Linearly increases the multiplicative variable from 0. to 1. over `warmup_steps` training steps.
        Keeps multiplicative variable equal to 1. after warmup_steps.
    """

    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupConstantSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return 1.

    @classmethod
    def build(cls, optimizer, cfg):
        if cfg.TRAINING.SCHEDULER.LEARNING_RATE_SCHEDULE_WARMUP_STEP_RATIO > 0:
            warmup_steps = cfg.TRAINING.SCHEDULER.LEARNING_RATE_SCHEDULE_WARMUP_STEP_RATIO * \
                           cfg.TRAINING.TRAIN_TOTAL_STEPS
        else:
            warmup_steps = 0.1 * cfg.TRAINING.TRAIN_TOTAL_STEPS
        warmup_steps = int(warmup_steps)
        return cls(optimizer, warmup_steps)


@plugin.register_plugin(plugin.PluginType.MODULE_SCHEDULER, 'warmup_linear')
class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Multiplies the learning rate defined in the optimizer by a dynamic variable determined by the current step.
        Linearly increases the multiplicative variable from 0. to 1. over `warmup_steps` training steps.
        Linearly decreases the multiplicative variable from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """

    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))

    @classmethod
    def build(cls, optimizer, cfg):
        if cfg.TRAINING.SCHEDULER.LEARNING_RATE_SCHEDULE_WARMUP_STEP_RATIO > 0:
            warmup_steps = cfg.TRAINING.SCHEDULER.LEARNING_RATE_SCHEDULE_WARMUP_STEP_RATIO * \
                           cfg.TRAINING.TRAIN_TOTAL_STEPS
        else:
            warmup_steps = 0.1 * cfg.TRAINING.TRAIN_TOTAL_STEPS
        warmup_steps = int(warmup_steps)
        return cls(optimizer, warmup_steps, cfg.TRAINING.TRAIN_TOTAL_STEPS)


# @plugin.register_plugin(plugin.PluginType.MODULE_SCHEDULER, 'warmup_cosine')
# class WarmupCosineSchedule(LambdaLR):
#     """ Linear warmup and then cosine decay.
#         Multiplies the learning rate defined in the optimizer by a dynamic variable determined by the current step.
#         Linearly increases the multiplicative variable from 0. to 1. over `warmup_steps` training steps.
#         Decreases the multiplicative variable from 1. to 0. over remaining `t_total - warmup_steps`
#         steps following a cosine curve.
#         If `cycles` (default=0.5) is different from default,
#         then the multiplicative variable follows cosine function after warmup.
#     """
#
#     def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
#         self.warmup_steps = warmup_steps
#         self.t_total = t_total
#         self.cycles = cycles
#         super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)
#
#     def lr_lambda(self, step):
#         if step < self.warmup_steps:
#             return float(step) / float(max(1.0, self.warmup_steps))
#         # progress after warmup
#         progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
#         return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
#
#     @classmethod
#     def build(cls, optimizer, config):
#         return cls(optimizer, config.lr_schedule_warmup_steps, config.lr_schedule_t_total)
#
#
# @plugin.register_plugin(plugin.PluginType.MODULE_SCHEDULER, 'warmup_cosine_with_hard_restarts')
# class WarmupCosineWithHardRestartsSchedule(LambdaLR):
#     """ Linear warmup and then cosine cycles with hard restarts.
#         Multiplies the learning rate defined in the optimizer by a dynamic variable determined by the current step.
#         Linearly increases the multiplicative variable from 0. to 1. over `warmup_steps` training steps.
#         If `cycles` (default=1.) is different from default, learning rate  follows `cycles` times a cosine decaying
#         learning rate (with hard restarts).
#     """
#
#     def __init__(self, optimizer, warmup_steps, t_total, cycles=1., last_epoch=-1):
#         self.warmup_steps = warmup_steps
#         self.t_total = t_total
#         self.cycles = cycles
#         super(WarmupCosineWithHardRestartsSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)
#
#     def lr_lambda(self, step):
#         if step < self.warmup_steps:
#             return float(step) / float(max(1, self.warmup_steps))
#         # progress after warmup
#         progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
#         if progress >= 1.0:
#             return 0.0
#         return max(0.0, 0.5 * (1. + math.cos(math.pi * ((float(self.cycles) * progress) % 1.0))))
#
#     @classmethod
#     def build(cls, optimizer, config):
#         return cls(optimizer, config.lr_schedule_warmup_steps, config.lr_schedule_t_total)
