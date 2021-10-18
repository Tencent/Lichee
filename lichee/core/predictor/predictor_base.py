# -*- coding: utf-8 -*-
import logging
import os
import shutil
import zipfile

from string import Template

import torch

from lichee import config
from lichee import plugin
from lichee.core import common
from lichee.utils import storage
from lichee.utils import sys_tmpfile
import json


@plugin.register_plugin(plugin.PluginType.PREDICTOR, "predictor_base")
class PredictorBase:
    def __init__(self, model_config_file):
        self.model_config_file = model_config_file
        config.merge_from_file(model_config_file)

        self.cfg = config.get_cfg()

        # gpu setting
        self.use_cuda = True
        self.master_gpu_id = 0
        self.gpu_ids = [0]
        self.init_gpu_setting()

        # init data model_loader
        self.eval_dataloader = None
        self.init_dataloader()

        # init model
        self.model = None
        self.init_model()

        # model inputs
        self.model_inputs = config.get_model_inputs()

        # init metrics
        self.metrics = []
        self.init_metrics()

        # init task cls
        self.task_cls = None
        self.init_task_cls()

        self.sample_inputs = None

        logging.info("predict config: %s", self.cfg)

    def init_gpu_setting(self):
        common.init_gpu_setting_default(self)

        if self.use_cuda:
            self.master_gpu_id = self.gpu_ids[0]

    def init_dataloader(self):
        common.init_dataloader_default(self)

    def init_model(self):
        model_cls = plugin.get_plugin(plugin.PluginType.UTILS_MODEL_LOADER, self.cfg.RUNTIME.EXPORT.TYPE)
        model_path = os.path.join(self.cfg.RUNTIME.SAVE_MODEL_DIR, self.cfg.RUNTIME.EXPORT.NAME)
        model_path = storage.get_storage_file(model_path)
        self.model = model_cls(model_path)
        logging.info("Start cpu training/evaluating...")

    def init_metrics(self):
        for metric_name in self.cfg.RUNTIME.METRICS.split(","):
            if metric_name == "none":
                continue
            metrics = plugin.get_plugin(plugin.PluginType.MODULE_METRICS, metric_name)
            self.metrics.append(metrics())

    def init_task_cls(self):
        self.task_cls = plugin.get_plugin(plugin.PluginType.TASK, self.cfg.MODEL.TASK.NAME)

    def predict(self):
        # self.eval_dataloader.dataset.is_training = False

        logging.info("predict start")
        is_write_head = False

        tmp_predict_res = sys_tmpfile.get_temp_file_path_once()
        f = open(tmp_predict_res, "w", encoding="utf-8")
        label_keys = []
        for step, batch in enumerate(self.eval_dataloader):
            inputs = self.get_inputs_batch(batch)
            label_keys, labels = self.get_label_batch(batch)
            label_vals = []
            with torch.no_grad():
                logits = self.model(inputs)

                model_outputs = self.task_cls.get_output(logits)
                if len(label_keys) != 0:
                    for label in labels:
                        label_vals.append(label.cpu().numpy())

            # get heads
            heads = self.get_result_heads(label_keys)

            # write heads once
            if not is_write_head:
                f.write("\t".join(heads) + "\n")
                is_write_head = True

            # get records
            record_arr = self.get_result_records(batch, label_vals, label_keys, model_outputs)

            if len(label_keys) > 0:
                for metric in self.metrics:
                    metric.collect(labels, logits)

            # write records
            for record in record_arr:
                f.write("\t".join(record) + "\n")

            if self.cfg.RUNTIME.DEBUG and (step + 1) > 4:
                break
        f.close()

        # upload predict result
        if "PREDICT" not in self.cfg.RUNTIME or self.cfg.RUNTIME.PREDICT.EXPORT_RESULT_PATH is None or \
                len(self.cfg.RUNTIME.PREDICT.EXPORT_RESULT_PATH) == 0:
            predict_res_path = os.path.join(self.cfg.RUNTIME.SAVE_MODEL_DIR, 'predict_res.txt')
        else:
            predict_res_path = self.cfg.RUNTIME.PREDICT.EXPORT_RESULT_PATH
        # storage
        storage.put_storage_file(tmp_predict_res, predict_res_path)

        if len(label_keys) > 0:
            for metric in self.metrics:
                metric.calc()

    def get_result_heads(self, label_keys: list):
        heads = []
        # head: original input
        for key in self.model_inputs:
            heads.append(key)

        # head: model output
        heads.append("prediction")

        # head: original label
        if len(label_keys) != 0:
            heads.extend(label_keys)
        return heads

    def get_result_records(self, batch, label_vals, label_keys, model_outputs):
        record_arr = []
        for i in range(model_outputs[0].shape[0]):
            record = []
            # value: original input
            for key in self.model_inputs:
                if "org_" + key in batch:
                    record.append(str(batch["org_" + key][i]))
                elif key in batch:
                    record.append(str(batch[key][i]))

            # value: model output
            outputs = []
            for item in model_outputs:
                outputs.append(item[i].tolist())
            record.append(json.dumps(tuple(outputs)))

            # value: original label
            if len(label_keys) != 0:
                for idx in range(len(label_keys)):
                    record.append(str(label_vals[idx][i]))
            record_arr.append(record)
        return record_arr

    def get_inputs_batch(self, batch):
        return common.get_inputs_batch_default(self, batch)

    def get_label_batch(self, batch):
        return common.get_label_batch_default(self, batch)
