# -*- coding: utf-8 -*-
import json
import logging
import os

import torch
from torch import distributed as dist
from torch.cuda.amp import GradScaler

from lichee import config
from lichee import plugin
from lichee.core import common as core_common
from lichee.dataset.dataloader import data_builder
from lichee.utils import common
from lichee.utils import parallel
from lichee.utils import storage
from lichee.utils import sys_tmpfile
from lichee.utils.convertor import torch_nn_convertor


@plugin.register_plugin(plugin.PluginType.TRAINER, "trainer_base")
class TrainerBase:
    def __init__(self, model_config_file, init_model=True):
        self.model_config_file = model_config_file
        self.train_dataloader = None
        self.eval_dataloader = None
        self.cfg = None
        self.use_cuda = None
        self.gpu_ids = None
        self.master_gpu_id = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss = None
        self.dist = False
        self.local_rank = 0
        self.rank = 0
        self.metrics = []
        self.eval_data = []
        self.temporary_map = {}  # save temporary values
        # init config
        self.init_config()
        # init seed
        self.init_seed()
        # init distributed
        self.init_distributed()
        # init dataloader
        self.init_dataloader()
        # init metrics
        self.init_metrics()
        if init_model:
            # init model
            self.init_model()
            # init optimize
            self.init_optimizer()
            # init schedule
            self.init_schedule()
            # init gpu setting
            self.init_gpu_setting()

        # model inputs
        self.model_inputs = config.get_model_inputs()
        # init autocast
        self.scaler = GradScaler()

    def init_config(self):
        config.merge_from_file(self.model_config_file)
        self.cfg = config.get_cfg()

    def init_seed(self):
        if "USE_SEED" in self.cfg.RUNTIME and self.cfg.RUNTIME.USE_SEED:
            common.set_seed(seed=self.cfg.RUNTIME.SEED)

    def init_distributed(self):
        implement = self.cfg.RUNTIME.IMPLEMENT
        # DP or DDP mode
        if implement == 'DistributedDataParallel':
            self.dist = True
            # dist params in cfg
            # init process group
            if 'DIST_PARAMS' in self.cfg.RUNTIME and len(self.cfg.RUNTIME.DIST_PARAMS) > 0:
                dist_params = self.cfg.RUNTIME.DIST_PARAMS
                if isinstance(dist_params, list):
                    parallel.init_dist(**parallel.get_dict_from_list(dist_params))
            else:
                parallel.init_dist()

    def init_gpu_setting(self):
        core_common.init_gpu_setting_default(self)

        if self.use_cuda:
            self.master_gpu_id = self.gpu_ids[0]
            if self.dist:
                # set local_rank and rank
                # default equal 0
                self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
                self.rank = dist.get_rank()
        if self.use_cuda and self.model is not None:
            if self.dist:
                # DDP mode
                self.model = self.model.cuda()
                logging.info("Start gpu distributed dataparallel training/evaluating...")
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank],
                                                                       output_device=self.local_rank)
            else:
                # DP mode
                self.model = self.model.cuda(self.master_gpu_id)
                logging.info("Start gpu dataparallel training/evaluating...")
                self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids)
        else:
            logging.info("Start cpu training/evaluating...")

    def gen_dataloader(self, data_config, training):
        data_path_list = []
        desc_data_path = storage.get_storage_file(self.cfg.DATASET.DESC_PATH)
        # get train file
        for train_data_path in data_config.DATA_PATH:
            train_file = storage.get_storage_file(train_data_path)
            train_files = [train_file]
            if "IS_FILE_LIST" in data_config and data_config["IS_FILE_LIST"]:
                train_files = [l.strip() for l in open(train_file, encoding='utf-8').readlines()]
            data_path_list.extend(train_files)
        desc_data_path = desc_data_path
        if 'DESC_PATH' in data_config:
            desc_data_path = storage.get_storage_file(data_config.DESC_PATH)
        if self.dist:
            # run with DDP mode
            dataloader = data_builder.build_ddp_dataloader(data_path_list,
                                                           desc_data_path,
                                                           self.use_cuda,
                                                           training=True,
                                                           shuffle=True)
        else:
            # run witn DP mode
            dataloader = data_builder.build_dataloader(data_config,
                                                       data_path_list,
                                                       desc_data_path,
                                                       self.use_cuda,
                                                       training=training,
                                                       shuffle=training)
        return dataloader

    def init_dataloader(self):
        self.train_dataloader = self.gen_dataloader(self.cfg.DATASET.TRAIN_DATA, training=True)
        if "EVAL_DATA" in self.cfg.DATASET:
            self.eval_dataloader = self.gen_dataloader(self.cfg.DATASET.EVAL_DATA, training=False)
        self.cfg.TRAINING.TRAIN_TOTAL_STEPS = len(self.train_dataloader) * self.cfg.TRAINING.EPOCHS

    def init_model(self):
        model_cls = plugin.get_plugin(plugin.PluginType.MODEL, self.cfg.MODEL.NAME)
        self.model = model_cls(self.cfg)

    def init_optimizer(self):
        optimize = plugin.get_plugin(plugin.PluginType.MODULE_OPTIMIZER, self.cfg.TRAINING.OPTIMIZER.NAME)
        self.optimizer = optimize.build(self.model, self.cfg)

    def init_schedule(self):
        schedule = plugin.get_plugin(plugin.PluginType.MODULE_SCHEDULER, self.cfg.TRAINING.SCHEDULER.NAME)
        self.scheduler = schedule.build(self.optimizer, self.cfg)

    def init_metrics(self):
        for metric_name in self.cfg.RUNTIME.METRICS.split(","):
            if metric_name == "none":
                continue
            metrics = plugin.get_plugin(plugin.PluginType.MODULE_METRICS, metric_name)
            self.metrics.append(metrics())

    def train(self):
        self.save_config_file()
        for epoch in range(1, self.cfg.TRAINING.EPOCHS + 1):
            logging.info("Training Epoch: " + str(epoch).center(60, "="))
            self.train_epoch()
            if self.eval_dataloader:
                self.eval_model(epoch)
                self.save_eval_data()
            self.save_model(epoch)

    def train_epoch(self):
        self.model.train()
        # self.train_dataloader.dataset.is_training = True

        for step, batch in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            inputs = self.get_inputs_batch(batch)
            label_keys, labels = self.get_label_batch(batch)
            for i in range(len(label_keys)):
                inputs[label_keys[i]] = labels[i]

            if self.cfg.RUNTIME.AUTOCAST:
                with torch.cuda.amp.autocast():
                    logits, loss = self.model(inputs)
            else:
                logits, loss = self.model(inputs)
            loss = loss.mean()
            if self.cfg.RUNTIME.AUTOCAST:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            if self.cfg.RUNTIME.AUTOCAST:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.scheduler.step()
            self.model.zero_grad()
            self.loss = loss.item()

            for metric in self.metrics:
                metric.collect(labels, logits)

            if (step + 1) % self.cfg.RUNTIME.REPORT_STEPS == 0:
                self.report_step(step)

            if self.cfg.RUNTIME.DEBUG and (step + 1) > 4:
                break
        for metric in self.metrics:
            metric.calc()

    def report_step(self, step):
        logging.info("Training step: %s, loss: %s", (step + 1), self.loss)

    def report_eval_step(self, report_info):
        logging.info("Eval epoch: {epoch}, loss: {loss}".format_map(report_info))

    def eval_model(self, epoch):
        if not self.eval_dataloader:
            logging.info('eval processing skipped no eval dataloader not initialized!')
            return
        epoch_eval_data = {"Key": "Epoch_" + str(epoch), 'epoch': epoch}
        self.eval_data.append(epoch_eval_data)
        self.model.eval()
        total_loss = 0.0
        num_batch = len(self.eval_dataloader)
        for step, batch in enumerate(self.eval_dataloader):
            inputs = self.get_inputs_batch(batch)
            label_keys, labels = self.get_label_batch(batch)
            for i in range(len(label_keys)):
                inputs[label_keys[i]] = labels[i]

            with torch.cuda.amp.autocast(), torch.no_grad():
                logits, loss = self.model(inputs)
                if isinstance(loss, torch.Tensor):
                    loss = loss.mean()
                loss_val = loss.item()
                total_loss += loss_val
                for metric in self.metrics:
                    metric.collect(labels, logits)

            if self.cfg.RUNTIME.DEBUG and (step + 1) > 4:
                break
        self.loss = total_loss / num_batch
        logging.info("Training Eval Metric".center(40, "-"))
        logging.info("Average Loss: " + format(total_loss / num_batch, "0.4f"))
        epoch_eval_data['loss'] = self.loss
        for metric in self.metrics:
            epoch_eval_data.update(metric.calc())
        self.report_eval_step(epoch_eval_data)

    def get_inputs_batch(self, batch):
        return core_common.get_inputs_batch_default(self, batch)

    def get_label_batch(self, batch):
        return core_common.get_label_batch_default(self, batch)

    def save_model(self, epoch):
        self.model.eval()
        # only storage by master
        if self.dist and self.rank != 0:
            return
        tmp_epoch_model_file = sys_tmpfile.get_temp_file_path_once()
        save_model_path = os.path.join(self.cfg.RUNTIME.SAVE_MODEL_DIR, "checkpoint", "Epoch_" + str(epoch) + '.bin')
        # decapsulation model
        torch_nn_convertor.TorchNNConvertor.save_model(self.model, None, tmp_epoch_model_file)
        storage.put_storage_file(tmp_epoch_model_file, save_model_path)

    def save_eval_data(self):
        # storage
        tmp_eval_data_file = sys_tmpfile.get_temp_file_path_once()
        open(tmp_eval_data_file, 'w', encoding='utf-8').write(json.dumps(self.eval_data))
        save_eval_data_path = os.path.join(self.cfg.RUNTIME.SAVE_MODEL_DIR, "res_info.json")
        storage.put_storage_file(tmp_eval_data_file, save_eval_data_path)

    def save_config_file(self):
        # storage
        save_config_file_path = os.path.join(self.cfg.RUNTIME.SAVE_MODEL_DIR, "task.yaml")
        storage.put_storage_file(self.model_config_file, save_config_file_path)
