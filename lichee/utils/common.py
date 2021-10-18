# -*- coding: utf-8 -*-
import collections
import logging
import time

from scipy.spatial.distance import cosine
import os
import random
from os.path import isfile, join
from functools import partial

import numpy as np
import torch

from lichee.utils import sys_tmpfile


def is_rank_0():
    return int(os.environ.get('RANK', 0)) == 0


def is_local_rank_0():
    return int(os.environ.get('LOCAL_RANK', 0)) == 0


def create_local_file_spinlock(filename: str):
    if is_local_rank_0() and not os.path.exists(filename):
        filename = os.path.split(filename)[-1]
        sys_tmpfile.create_temp_lock_file(filename)
        return True
    return False


def get_local_file_spinlock(filename: str):
    while not is_local_rank_0() and not sys_tmpfile.exist_temp_lock_file(filename):
        time.sleep(0.1)
        continue
    return True


# 随机数固定，RE-PRODUCIBLE
def set_seed(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_saved_model(model, saved_model_path, model_file=None):
    # If file not defined, load last saved file by default
    if model_file == None:
        files = get_files_sort_by_mtime(saved_model_path, reverse=True)
        model_file = files[0]
    else:
        model_file = os.path.join(saved_model_path, model_file)
    load_model_dict(model, model_file)


def load_model_dict(model, model_file):
    model_weight = torch.load(model_file, map_location="cpu")
    new_state_dict = collections.OrderedDict()
    for k, v in model_weight.items():
        if k.startswith("module"):
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)


def get_files_sort_by_mtime(file_dir, fname_key='', reverse=True):
    file_paths = {}
    for f in os.listdir(file_dir):
        if isfile(join(file_dir, f)) and fname_key in f:
            file_path = join(file_dir, f)
            mtime = os.path.getmtime(file_path)
            file_paths[file_path] = mtime
    file_paths = {k: v for k, v in sorted(file_paths.items(), key=lambda item: item[1], reverse=reverse)}
    return list(file_paths.keys())


def save_model(save_model_path, model, model_name):
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)
    model_weight = model.state_dict()
    new_state_dict = collections.OrderedDict()
    for k, v in model_weight.items():
        if k.startswith("module"):
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    model_dir = os.path.join(save_model_path, model_name)
    torch.save(new_state_dict, model_dir)


def load_word2vec_vector(vector_file, vector_dim):
    vectors = np.fromfile(vector_file, dtype=np.float32)
    vectors = vectors.reshape([-1, vector_dim])

    # logging.info('word embedding init'.center(60, '='))
    # logging.info(vectors[:10])
    # logging.info('vec shape:%s', vectors.shape)
    return vectors


def output_metric(test_outputs, ref_outputs, output_count):
    max_diff = [0.0 for _ in range(output_count)]
    max_l1_norm = [0.0 for _ in range(output_count)]
    max_l2_norm = [0.0 for _ in range(output_count)]
    max_cos_distance = [0.0 for _ in range(output_count)]

    for output, ref_output in zip(test_outputs, ref_outputs):
        for i in range(output_count):
            lhs = output[i].flatten()
            rhs = ref_output[i].flatten()
            assert lhs.size == rhs.size

            max_diff[i] = max(max_diff[i], np.max(np.abs(lhs - rhs)))
            max_l1_norm[i] = max(max_l1_norm[i], np.sum(np.abs(lhs - rhs)))
            max_l2_norm[i] = max(max_l2_norm[i], np.sum((lhs - rhs) ** 2))
            max_cos_distance[i] = max(max_cos_distance[i], cosine(lhs, rhs))

    for i in range(output_count):
        logging.info('Output %d: shape %s, dtype %s' % (i, test_outputs[0][i].shape, test_outputs[0][i].dtype))
        logging.info('max difference %.7f, L1 norm %.7f, L2 norm %.7f, cosine difference %.7f' %
                     (max_diff[i], max_l1_norm[i], max_l2_norm[i], max_cos_distance[i]))

    return np.max(max_diff)


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.
    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.
    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))
