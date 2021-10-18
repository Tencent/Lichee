# -*- coding: utf-8 -*-
from lichee import plugin
from .field_parser_base import BaseFieldParser
import os
from PIL import Image
from torchvision import transforms
import torch
from lichee.utils import storage


@plugin.register_plugin(plugin.PluginType.FIELD_PARSER, "image_local_path")
class ImgDataFieldParser(BaseFieldParser):
    """The field parser for local image. Read the image data from the path provided,
    transforms through ToSensor, Resize and Normalize.

    Attributes
    ----------
    transformer: transforms.Compose
        compose the transforms(ToSensor, Resize and Normalize)
    """
    def __init__(self):
        super().__init__()
        self.transformer = None

    def init(self, cfg):
        self.cfg = cfg
        resolution = [int(x) for x in self.global_config.DATASET.CONFIG.IMAGE_RESOLUTION]
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resolution),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def parse(self, row, training=False):
        """Parse the row and obtain the path of image, invoke prepare_img_data to transform the image data to tensor.

        Parameters
        ----------
        row: memoryview
            Object contained in a single record
        training: bool
            inherited from parent, not used here.

        Returns
        -------
        record: torch.Tensor
            the tensor of image data
        """
        record = {}
        if self.key not in row:
            raise Exception("Cannot find key %s in row by image_local_path" % self.key)

        img_path = bytes(row[self.key]).decode("utf-8")
        if img_path[0] != "/":
            img_path = os.path.join(self.global_config.DATASET.DATA_BASE_DIR, img_path)
        record[self.alias] = self.prepare_img_data(img_path)
        return record

    def prepare_img_data(self, img_path):
        """Read and process the image from image_path

        Parameters
        ----------
        img_path: str
            path of image

        Returns
        ------
        torch.Tensor
            the tensor transformed from image data.

        """
        with open(storage.get_storage_file(img_path), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            return self.transformer(img)

    def collate(self, batch):
        record = {}
        imgs = [instance[self.alias] for instance in batch]
        imgs = torch.stack(imgs)
        record[self.alias] = imgs
        return record
