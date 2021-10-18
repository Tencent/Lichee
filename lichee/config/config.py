# -*- coding: utf-8 -*-
import os

from yacs.config import CfgNode as CN

# --------------------------------------------------------------------------- #
# Root handle for all configs
# --------------------------------------------------------------------------- #
_C = CN()

# --------------------------------------------------------------------------- #
# Dataset settings, including data abstractions, contents and transforms
# --------------------------------------------------------------------------- #
_C.DATASET = CN()
_C.DATASET.TYPE = ""
_C.DATASET.NAME = ""
_C.DATASET.FORMAT = ""
_C.DATASET.FIELD = []
_C.DATASET.DESC_PATH = ""
_C.DATASET.TRAIN_DATA = CN()
_C.DATASET.TRAIN_DATA.DATA_PATH = []
_C.DATASET.TRAIN_DATA.BATCH_SIZE = 0
_C.DATASET.TRAIN_DATA.WORKER_NUM = 0
_C.DATASET.EVAL_DATA = CN()
_C.DATASET.EVAL_DATA.DATA_PATH = []
_C.DATASET.EVAL_DATA.BATCH_SIZE = 0
_C.DATASET.EVAL_DATA.WORKER_NUM = 0
_C.DATASET.CONFIG = CN()

# --------------------------------------------------------------------------- #
# Model configs, including backbone, neck, multiple heads and extend for others
# --------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.TYPE = ""
_C.MODEL.NAME = ""
_C.MODEL.REPRESENTATION = []
_C.MODEL.TASK = CN()
_C.MODEL.TASK.NAME = ""
_C.MODEL.TASK.CONFIG = CN()
_C.MODEL.TASK.CONFIG.LOSS = CN()
_C.MODEL.TASK.CONFIG.LOSS.NAME = ""
_C.MODEL.TASK.CONFIG.LOSS.VALUE_TYPE = ""
_C.MODEL.TASK.CONFIG.LOSS.PARAM = CN()
_C.MODEL.GRAPH = []
_C.MODEL.CONFIG = CN()

# ---------------------------------------------------------------------------- #
# Runtime settings, including hooks, logs, evaluations and so on
# ---------------------------------------------------------------------------- #
_C.RUNTIME = CN()
_C.RUNTIME.IMPLEMENT = ""
_C.RUNTIME.LOG_LEVEL = ""
_C.RUNTIME.SEED = 0
_C.RUNTIME.GPU_IDS = ""
_C.RUNTIME.SAVE_MODEL_DIR = ""
_C.RUNTIME.EXPORT = CN()
_C.RUNTIME.EXPORT.TYPE = ""
_C.RUNTIME.EXPORT.NAME = ""
_C.RUNTIME.DEBUG = False
_C.RUNTIME.REPORT_STEPS = 0
_C.RUNTIME.METRICS = ""
_C.RUNTIME.CONFIG = CN()

# ---------------------------------------------------------------------------- #
# Schedule methods, including optimizers and learning policies
# ---------------------------------------------------------------------------- #
_C.TRAINING = CN()
_C.TRAINING.EPOCHS = 0

_C.TRAINING.OPTIMIZER = CN()
_C.TRAINING.OPTIMIZER.NAME = ""
_C.TRAINING.OPTIMIZER.LEARNING_RATE = 0.1
_C.TRAINING.OPTIMIZER.OPTIM_WEIGHT_DECAY = 1e-4
_C.TRAINING.OPTIMIZER.OPTIM_EPS = 1e-8
_C.TRAINING.OPTIMIZER.OPTIM_MOMENTUM = 0.9
_C.TRAINING.OPTIMIZER.OPTIM_BETA1 = 0.9
_C.TRAINING.OPTIMIZER.OPTIM_BETA2 = 0.999
_C.TRAINING.OPTIMIZER.OPTIM_AMSGRAD = False
_C.TRAINING.OPTIMIZER.CORRECT_BLAS = True
_C.TRAINING.OPTIMIZER.LEARNING_RATE_SCHEDULE = ""
_C.TRAINING.OPTIMIZER.GRADIENT_ACCUMULATION_STEPS = 1
_C.TRAINING.OPTIMIZER.MAX_GRAD_NORM = 0.0
_C.TRAINING.OPTIMIZER.GRAD_NORM_TYPE = 0.0
_C.TRAINING.OPTIMIZER.MAX_GRAD_VALUE = 0.0

_C.TRAINING.SCHEDULER = CN()
_C.TRAINING.SCHEDULER.NAME = ""
_C.TRAINING.SCHEDULER.LEARNING_RATE_SCHEDULE_WARMUP_STEP_RATIO = 0.0

_C.set_new_allowed(True)


def get_cfg():
    return _C


def merge_from_file(file_path):
    cfg = get_cfg()
    cfg.merge_from_file(file_path)


def freeze():
    cfg = get_cfg()
    cfg.freeze()


def init_cfg():
    cur_path = os.path.abspath(__file__)
    cur_dir = os.path.split(cur_path)[0]
    merge_from_file(os.path.join(cur_dir, '_base_/datasets/dataset.yaml'))
    merge_from_file(os.path.join(cur_dir, '_base_/models/model.yaml'))
    merge_from_file(os.path.join(cur_dir, '_base_/runtimes/runtime.yaml'))
    merge_from_file(os.path.join(cur_dir, '_base_/training/training.yaml'))


init_cfg()


def get_model_inputs():
    cfg = get_cfg()

    fields = set()
    for field in cfg.DATASET.FIELD:
        keys_str = field["ALIAS"] if field.get("ALIAS", None) else field["KEY"]
        if not keys_str:
            continue
        fields.update([keys_str])
    model_inputs_set = set()
    for node in cfg.MODEL.GRAPH:
        model_inputs_set.update(set(node["INPUTS"]).intersection(fields))
    return list(model_inputs_set)
