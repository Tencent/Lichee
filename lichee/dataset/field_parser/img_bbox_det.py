# -*- coding: utf-8 -*-
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from lichee import plugin
from .field_parser_base import BaseFieldParser


@plugin.register_plugin(plugin.PluginType.FIELD_PARSER, "img_bbox_det")
class ImgBBoxDetFieldParser(BaseFieldParser):
    """
    field parser for image detection

    """
    def __init__(self):
        super().__init__()
        self.inner_key = ['image', 'gt_bboxes', 'gt_labels']
        self.transformer = None
        self.min_size = 5

    def init(self, cfg):
        self.cfg = cfg
        self.img_shape = [int(x) for x in self.global_config.DATASET.CONFIG.IMAGE_RESOLUTION]
        img_mean = [float(x) for x in self.global_config.DATASET.CONFIG.IMAGE_MEAN]
        img_std = [float(x) for x in self.global_config.DATASET.CONFIG.IMAGE_STD]
        self.transformer = transforms.Compose([
            transforms.Resize(self.img_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std)
        ])

    def _resize_bboxes(self, ori_bboxes, scale_factor):
        """
        Resize bounding boxes with scale_factor.
        """
        bboxes = ori_bboxes * scale_factor
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, self.img_shape[1])
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, self.img_shape[0])
        return bboxes

    def parse(self, row, training=False):
        """
        transfer original inputs to model inputs ('image', 'gt_bboxes', 'gt_labels')

        :param row: Dict
        :param training: bool
        :return: Dict
        """
        record = {}
        for alias, key in self.key_map.items():
            if key not in row:
                continue
            if key == 'image':
                image_raw = row[key]
                pil_img = Image.open(BytesIO(image_raw)).convert('RGB')
                img_tensor = self.transformer(pil_img)

            elif key == 'bbox/class':
                obj_cls = row[key]
            elif key == 'bbox/xmin':
                obj_xmin = row[key]
            elif key == 'bbox/ymin':
                obj_ymin = row[key]
            elif key == 'bbox/xmax':
                obj_xmax = row[key]
            elif key == 'bbox/ymax':
                obj_ymax = row[key]

        bboxes = []
        labels = []

        for i in range(len(obj_cls)):
            label = obj_cls[i]
            bbox = [
                float(obj_xmin[i]),
                float(obj_ymin[i]),
                float(obj_xmax[i]),
                float(obj_ymax[i])
            ]

            ignore = False
            if self.min_size:
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True

            if not ignore:
                bboxes.append(bbox)
                labels.append(label)

        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0,))
        else:
            bboxes = np.array(bboxes, ndmin=2)
            labels = np.array(labels).astype(np.int64)

        width = pil_img.size[0]
        height = pil_img.size[1]

        h_scale = 1.0 * self.img_shape[0] / height
        w_scale = 1.0 * self.img_shape[1] / width

        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)

        bboxes = self._resize_bboxes(bboxes, scale_factor)

        record['image'] = img_tensor
        record['gt_bboxes'] = torch.from_numpy(bboxes)
        record['gt_labels'] = torch.from_numpy(labels)

        return record

    def collate(self, batch):
        record = {}
        for key in self.inner_key:
            if key == 'image':
                datas = [instance[key] for instance in batch]
                datas = torch.stack(datas)
                record[key] = datas
            else:
                record[key] = [sample[key] for sample in batch]

        return record
