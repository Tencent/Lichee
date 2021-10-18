# -*- coding: utf-8 -*-
import json
import logging
import os

import numpy as np
import scipy
import torch
import tqdm

from lichee import plugin
from lichee.core.trainer.trainer_base import TrainerBase
from lichee.utils import storage
from lichee.utils import sys_tmpfile
from lichee.utils.convertor import torch_nn_convertor


def float_to_str(float_list):
    return ','.join(['%f' % val for val in float_list])


@plugin.register_plugin(plugin.PluginType.TASK, 'concat_cls')
class ConcatCls(torch.nn.Module):
    def __init__(self, cfg):
        '''
        :param cfg: concat_cls config defined in your_config.yaml
        '''
        super().__init__()
        self.cfg = cfg
        self.fc_hidden = torch.nn.Linear(cfg['INPUT_SIZE'], cfg['HIDDEN_SIZE'])
        self.fc_logits = torch.nn.Linear(cfg['HIDDEN_SIZE'], cfg['NUM_CLASSES'])
        self.loss_func = None
        self.init_loss()

    def forward(self, video_feature, title_feature, label=None):
        '''
        :param video_feature: video feature extracted from frame representation
        :param title_feature: title feature extracted from title representation
        :param label: classification target
        :return: (predictions, embeddings), model loss
        '''
        title_feature = title_feature[:, 0]
        feature = torch.cat([video_feature, title_feature], dim=1)
        embedding = self.fc_hidden(feature)
        normed_embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        pred = self.fc_logits(torch.relu(embedding))
        loss = None
        if label is not None:
            label = label[self.cfg["LABEL_KEY"]].float()
            loss = self.loss_func(pred, label)
            if 'SCALE' in self.cfg['LOSS']:
                loss = loss * self.cfg['LOSS']['SCALE']
        pred = torch.sigmoid(pred)
        return (pred, normed_embedding), loss

    def init_loss(self):
        loss = plugin.get_plugin(plugin.PluginType.MODULE_LOSS, self.cfg['LOSS']['NAME'])
        self.loss_func = loss.build(self.cfg['LOSS'])


@plugin.register_plugin(plugin.PluginType.TRAINER, 'embedding_trainer')
class EmbeddingTrainer(TrainerBase):

    def __init__(self, config, init_model=True):
        '''
        :param config: the global trainer config, defined by your_config.yaml
        :param init_model: whether initialize the model or not
        '''
        super().__init__(config, init_model)

    def report_step(self, step):
        metric = self.metrics[0]
        metric_info = metric.calc()
        metric_info['loss'] = self.loss
        metric_info['step'] = step
        logging.info(
            "Step {step}, precision: {precision:.4f}, recall: {recall:.4f}, loss: {loss:.4f}".format_map(metric_info))

    def report_eval_step(self, metric_info):
        print(metric_info)
        print(type(metric_info))
        logging.info(
            "EVAL EPOCH {epoch}, precision: {precision:.4f}, recall: {recall:.4f}, loss: {loss:.4f}".format_map(
                metric_info))
        self.temporary_map.update(metric_info)

    def evalute_checkpoint(self, checkpoint_file: str, dataset_key: str, to_save_file):
        '''
        :param checkpoint_file: the checkpoint used to evalutate the dataset
        :param dataset_key: dataset indicator key, defined by your_config.yaml DATASET block
        :param to_save_file: the file to save the result
        :return:
        '''
        assert dataset_key in self.cfg.DATASET
        dataset_config = self.cfg.DATASET[dataset_key]
        dataset_loader = self.gen_dataloader(dataset_config, training=False)
        self.load_checkpoint_for_eval(checkpoint_file)
        self.model.eval()
        epoch_start = checkpoint_file.find("_")
        epoch_end = checkpoint_file.find("_", epoch_start + 1)
        self.eval_model(int(checkpoint_file[epoch_start + 1 : epoch_end]))

    def evaluate_spearman(self, checkpoint_file='', dataset_key="SPEARMAN_EVAL"):
        '''
        :param checkpoint_file: the checkpoint used to evalutate the dataset
        :param dataset_key: dataset indicator key, defined by your_config.yaml DATASET block
        :return:
        '''
        if checkpoint_file:
            self.load_checkpoint_for_eval(checkpoint_file)
        self.model.eval()
        dataset_config = self.cfg.DATASET[dataset_key]
        dataset_loader = self.gen_dataloader(dataset_config, training=False)
        id_list = []
        embedding_list = []
        for step, batch in tqdm.tqdm(enumerate(dataset_loader)):
            inputs = self.get_inputs_batch(batch)
            ids = batch['id']
            (logits, embedding), _ = self.model(inputs)
            embedding = embedding.detach().cpu().numpy()
            embedding_list.append(embedding)
            id_list += ids
        embeddings = np.concatenate(embedding_list)
        embedding_map = dict(zip(id_list, embeddings))
        annotate = {}
        label_file = storage.get_storage_file(dataset_config['LABEL_FILE'])
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                rk1, rk2, score = line.split('\t')
                annotate[(rk1, rk2)] = float(score)
        sim_res = []
        logging.info('num embedding: {}, num annotates: {}'.format(len(embedding_map), len(annotate)))
        for (k1, k2), v in annotate.items():
            if k1 not in embedding_map or k2 not in embedding_map:
                continue
            sim_res.append((v, (embedding_map[k1] * embedding_map[k2]).sum()))
        spearman = scipy.stats.spearmanr([x[0] for x in sim_res], [x[1] for x in sim_res]).correlation
        logging.info('spearman score: {}'.format(spearman))
        self.temporary_map['spearman'] = spearman

    def save_model(self, epoch):
        self.model.eval()
        self.temporary_map.update(self.eval_data[-1])  # update temporay_map with lattest eval info
        tmp_epoch_model_file = sys_tmpfile.get_temp_file_path_once()
        save_model_path = os.path.join(self.cfg.RUNTIME.SAVE_MODEL_DIR, "checkpoint",
                                       "Epoch_{epoch}_{precision:.4f}_{recall:.4f}.bin".format_map(
                                           self.temporary_map))
        torch_nn_convertor.TorchNNConvertor.save_model(self.model, None, tmp_epoch_model_file)
        storage.put_storage_file(tmp_epoch_model_file, save_model_path)

    def train(self):
        self.save_config_file()
        for epoch in range(1, self.cfg.TRAINING.EPOCHS + 1):
            logging.info("Training Epoch: " + str(epoch).center(60, "="))
            self.train_epoch()
            if 'SPEARMAN_EVAL' in self.cfg.DATASET:  # run spearnman test if eval if SPEARMAN_EVAL config is found
                self.evaluate_spearman(dataset_key='SPEARMAN_EVAL')
            if self.eval_dataloader:  # run eval
                self.eval_model(epoch)
            self.save_model(epoch)

    def load_checkpoint_for_eval(self, checkpoint_file):
        '''
        :param checkpoint_file: checkpoint file used to eval model
        :return:
        '''
        save_model_dir = os.path.join(self.cfg.RUNTIME.SAVE_MODEL_DIR, "checkpoint")
        model_path = os.path.join(save_model_dir, checkpoint_file)
        model_file = storage.get_storage_file(model_path)
        self.model = torch.load(model_file)
        self.init_gpu_setting()
        self.model.eval()

    def empty_loop_test(self):
        '''
        :return: empty loop to test IO speed
        '''
        for _ in tqdm.tqdm(self.train_dataloader):
            continue
