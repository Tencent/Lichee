# -*- coding: utf-8 -*-
import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer

from lichee import plugin
from lichee.dataset.field_parser.field_parser_base import BaseFieldParser
from lichee.utils import storage


@plugin.register_plugin(plugin.PluginType.FIELD_PARSER, 'frame_feature')
class FrameFeature(BaseFieldParser):
    def __init__(self):
        super().__init__()
        self.zero_frame = None
        self.num_segments = None

    def init(self, cfg):
        '''
        :param cfg: frame feature config, defined in your_config.yaml
        :return:
        '''
        self.cfg = cfg
        self.num_segments = cfg['NUM_SEGMENT']

    def parse(self, row, training=False):
        '''
        :param row: raw feature map {key1: raw_feature1, key2: raw_feature2, ...}
        :param training: training or not can behave different
        :return: frame_feature with {self.alias: feature}
        '''
        record = {}
        frame_feature = [np.frombuffer(bytes(x), dtype=np.float16).astype(np.float32) for x in row[self.alias]]
        if self.zero_frame is None:
            self.zero_frame = frame_feature[0] * 0.
        num_frames = len(frame_feature)
        frame_gap = (num_frames - 1) / self.num_segments
        if frame_gap <= 1:
            res = frame_feature + [self.zero_frame] * (self.num_segments - num_frames)
        else:
            if training:
                res = [frame_feature[round(i * frame_gap + np.random.uniform(0, frame_gap))] for i in
                       range(self.num_segments)]
            else:
                res = [frame_feature[round((i + 0.5) * frame_gap)] for i in range(self.num_segments)]
        record[self.alias] = torch.tensor(np.c_[res])
        return record

    def collate(self, batch):
        '''
        :param batch: list of items samples like [{key1: feature1, ...},  {key1: feature1, ...}, ...]
        :return: batched data like {key1: batch_feature1, key2: batch_feature2, ...}
        '''
        record = {}
        batch_record = [instance[self.alias] for instance in batch]
        batch_frame_feature = torch.cat([torch.unsqueeze(f, 0) for f in batch_record])
        record[self.alias] = batch_frame_feature
        return record


@plugin.register_plugin(plugin.PluginType.FIELD_PARSER, 'tag_cls')
class TagParser(BaseFieldParser):
    def __init__(self):
        super().__init__()
        self.selected_tags = set()
        self.mlb = MultiLabelBinarizer()
        self.tag_trans_map = None

    def init(self, cfg):
        '''
        :param cfg: tag config from your_config.yaml
        :return:
        '''
        self.cfg = cfg
        tag_file = storage.get_storage_file(cfg['TAG_FILE'])
        with open(tag_file, encoding='utf-8') as fh:
            for line in fh:
                fields = line.strip().split('\t')
                self.selected_tags.add(int(fields[0]))
        assert len(self.selected_tags) == cfg['TAG_SIZE']
        self.mlb.fit([self.selected_tags])

    def parse(self, row, training=False):
        '''
        :param row: raw feature map {key1: raw_feature1, key2: raw_feature2, ...}
        :param training: training or not can behave different
        :return: tag_cls feature with {self.alias: feature}
        '''
        record = {}
        if self.alias not in row:
            return None
        tags = row[self.alias]
        tags = [t for t in tags if t in self.selected_tags]
        multi_hot = self.mlb.transform([tags])[0]
        record[self.alias] = torch.LongTensor(multi_hot)
        return record

    def collate(self, batch):
        '''
        :param batch: list of items samples like [{key1: feature1, ...},  {key1: feature1, ...}, ...]
        :return: batched data like {key1: batch_feature1, key2: batch_feature2, ...}
        '''
        record = {}
        if any([self.alias not in instance for instance in batch]):
            return None
        batch_record = [instance[self.alias] for instance in batch]
        batch_frame_feature = torch.cat([torch.unsqueeze(f, 0) for f in batch_record])
        record[self.alias] = batch_frame_feature
        return record


@plugin.register_plugin(plugin.PluginType.FIELD_PARSER, 'id')
class VidParser(BaseFieldParser):

    def parse(self, row, training=False):
        '''
        :param row: raw feature map {key1: raw_feature1, key2: raw_feature2, ...}
        :param training: training or not can behave different
        :return: id feature with {self.alias: feature}
        '''
        record = {}
        vid = [bytes(row['id']).decode('utf-8')]
        record[self.alias] = vid
        return record

    def collate(self, batch):
        '''
        :param batch: list of items samples like [{key1: feature1, ...},  {key1: feature1, ...}, ...]
        :return: batched data like {key1: batch_feature1, key2: batch_feature2, ...}
        '''
        record = dict()
        record[self.alias] = []
        for instance in batch:
            record[self.alias].extend(instance[self.alias])
        return record
