# -*- coding: utf-8 -*-
import copy
import logging
import os

import torch

from lichee import config
from lichee import plugin
from lichee.core import common
from lichee.utils import storage
from lichee.utils import sys_tmpfile
from lichee.utils.convertor.convertor_base import BaseConvertor


@plugin.register_plugin(plugin.PluginType.EVALUATOR, "evaluator_base")
class EvaluatorBase:
    def __init__(self, model_config_file):
        config.merge_from_file(model_config_file)

        self.cfg = config.get_cfg()

        # gpu setting
        self.use_cuda = True
        self.master_gpu_id = 0
        self.gpu_ids = [0]
        self.init_gpu_setting()

        # model inputs
        self.model = None
        self.model_inputs = config.get_model_inputs()

        # init data model_loader
        self.eval_dataloader = None
        self.init_dataloader()

        # init metrics
        self.metrics = []
        self.init_metrics()

        # set trace or sample
        self.trace_inputs = None
        self.sample_inputs = None

        logging.info("evaluate config: %s", self.cfg)

    def init_gpu_setting(self):
        common.init_gpu_setting_default(self)

        if self.use_cuda:
            self.master_gpu_id = self.gpu_ids[0]

    def init_dataloader(self):
        common.init_dataloader_default(self)

    def init_metrics(self):
        for metric_name in self.cfg.RUNTIME.METRICS.split(","):
            if metric_name == "none":
                continue
            # metric plugin
            metrics = plugin.get_plugin(plugin.PluginType.MODULE_METRICS, metric_name)
            self.metrics.append(metrics())

    def eval(self):
        eval_dict = {}
        model_file_dict = {}
        save_model_dir = os.path.join(self.cfg.RUNTIME.SAVE_MODEL_DIR, "checkpoint")
        for epoch in range(1, self.cfg.TRAINING.EPOCHS + 1):
            model_name = "Epoch_" + str(epoch) + '.bin'
            model_path = os.path.join(save_model_dir, model_name)
            # get model file
            model_file = storage.get_storage_file(model_path)
            # load model
            self.model = torch.load(model_file)
            loss, acc = self.eval_model()
            eval_dict[model_path] = acc
            model_file_dict[model_path] = model_file

        eval_dict = {k: v for k, v in sorted(eval_dict.items(), key=lambda item: item[1], reverse=True)}
        best_model_path = list(eval_dict.keys())[0]

        logging.info("Eval Compare".center(60, "="))
        logging.info(eval_dict)
        logging.info("choose best model:%s", best_model_path)
        # load
        self.model = torch.load(model_file_dict[best_model_path])

        # export
        export_model_path = os.path.join(self.cfg.RUNTIME.SAVE_MODEL_DIR, self.cfg.RUNTIME.EXPORT.NAME)
        tmp_convert_model_file = sys_tmpfile.get_temp_file_path_once()
        convertor_cls: BaseConvertor = plugin.get_plugin(plugin.PluginType.UTILS_CONVERTOR,
                                                         self.cfg.RUNTIME.EXPORT.TYPE)
        # storage
        convertor_cls.convert(self.model, self.trace_inputs, self.sample_inputs, tmp_convert_model_file)
        storage.put_storage_file(tmp_convert_model_file, export_model_path)
        # clear temp file
        os.remove(tmp_convert_model_file)

    def eval_model(self):
        self.model.eval()
        # self.eval_dataloader.dataset.is_training = False

        total_loss = 0.0
        num_batch = self.eval_dataloader.__len__()
        for step, batch in enumerate(self.eval_dataloader):
            inputs = self.get_inputs_batch(batch)
            if self.trace_inputs is None:
                self.trace_inputs = copy.deepcopy(inputs)
            self.sample_inputs = copy.deepcopy(inputs)

            label_key, labels = self.get_label_batch(batch)
            if isinstance(label_key, list):
                for i in range(len(label_key)):
                    inputs[label_key[i]] = labels[i]
            else:
                inputs[label_key] = labels

            with torch.cuda.amp.autocast(), torch.no_grad():
                logits, loss = self.model(inputs)
                loss = loss.mean()
                loss_val = loss.item()
                total_loss += loss_val

                for metric in self.metrics:
                    metric.collect(labels, logits)

            if (step + 1) % self.cfg.RUNTIME.REPORT_STEPS == 0:
                logging.info("eval step: %s ", step + 1)

            if self.cfg.RUNTIME.DEBUG and (step + 1) > 4:
                break

        loss_v = total_loss / num_batch
        logging.info("Average Loss: " + format(loss_v, "0.4f"))

        acc = 0
        for i, metric in enumerate(self.metrics):
            acc = i

            if metric.name() == "Accuracy":
                acc = metric.calc()["Acc"]["value"]
                break

        logging.info("Accuracy: " + format(acc, "0.4f"))

        return loss_v, acc

    def get_inputs_batch(self, batch):
        return common.get_inputs_batch_default(self, batch)

    def get_label_batch(self, batch):
        return common.get_label_batch_default(self, batch)
