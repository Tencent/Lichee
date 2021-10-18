# -*- coding: utf-8 -*-
import io

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from lichee import plugin
from .field_parser_base import BaseFieldParser


@plugin.register_plugin(plugin.PluginType.FIELD_PARSER, "video_tsn")
class VideoTemporalSegmentSampleParser(BaseFieldParser):
    """field parser for video TSN model

    Attributes
    ----------
    image_resolution: Tuple[int]
        image resolution in preprocess
    trans: transforms.Compose
        image preprocess transform

    """
    def __init__(self):
        super().__init__()

        self.image_resolution = None
        self.trans = None

    def init(self, cfg):
        self.cfg = cfg
        self.image_resolution = [int(x) for x in self.cfg.DATASET.CONFIG.IMAGE_RESOLUTION]
        self.trans = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize(self.image_resolution),
                                         transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])])

    def parse(self, row, training=False):
        record = {}
        if self.key not in row:
            raise Exception("Cannot find key %s in row by video_tsn" % self.key)

        frame_bytes = [f.tobytes() for f in row[self.key]]
        frame_bytes, frame_num = self._sample(frame_bytes, self.cfg.DATASET.CONFIG.NUM_SEGMENTS, training)
        frames = [Image.open(io.BytesIO(frame_bytes)) for frame_bytes in frame_bytes]

        frames = [self.trans(f) for f in frames]
        frames = torch.cat([torch.unsqueeze(f, 0) for f in frames])
        record[self.alias] = frames

        return record

    def collate(self, batch):
        record = {}
        batch_frames = [instance[self.alias] for instance in batch]
        batch_frames = torch.cat([torch.unsqueeze(f, 0) for f in batch_frames])
        record[self.alias] = batch_frames
        return record

    @staticmethod
    def _sample(frames, num_segments, training):
        frames_len = len(frames)
        num_frames = min(frames_len, num_segments)
        average_duration = frames_len // num_segments
        if average_duration == 0:
            return [frames[min(i, frames_len - 1)] for i in range(num_segments)], np.array([num_frames], dtype=np.int32)
        else:
            if training:
                offsets = np.multiply(list(range(num_segments)), average_duration) + \
                          np.random.randint(average_duration, size=num_segments)
            else:
                offsets = np.multiply(list(range(num_segments)), average_duration) + average_duration // 2
        return [frames[i] for i in offsets], np.array([num_frames], dtype=np.int32)
