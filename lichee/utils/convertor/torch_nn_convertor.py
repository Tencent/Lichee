# -*- coding: utf-8 -*-
import logging

import torch

from lichee.utils import common
from lichee import plugin
from lichee.utils.convertor import convertor_base


@plugin.register_plugin(plugin.PluginType.UTILS_CONVERTOR, "torch_nn")
class TorchNNConvertor(convertor_base.BaseConvertor):
    """provides ability to convert and save model to torch nn format

    """

    @classmethod
    def convert(cls, inited_model, trace_inputs, sample_inputs, export_model_path):
        """convert and save the model

        Parameters
        ----------
        inited_model: torch.nn.Module
            model to save
        trace_inputs: object
            sample inputs used to convert to model
        sample_inputs: object
            sample inputs used to check converted model
        export_model_path: str
            path to save exported model

        """
        cls.save_model(inited_model, trace_inputs, export_model_path)
        cls.check_and_infer(inited_model, sample_inputs, export_model_path)

    @classmethod
    def save_model(cls, inited_model, trace_inputs, export_model_path):
        if isinstance(inited_model, torch.nn.DataParallel):
            inited_model = inited_model.module
        elif isinstance(inited_model, torch.nn.parallel.DistributedDataParallel):
            inited_model = inited_model.module

        inited_model.eval()

        torch.save(inited_model, export_model_path)

    @classmethod
    def check_and_infer(cls, inited_model, valid_data, export_model_path):
        model = torch.load(export_model_path)

        # infer
        org_outputs = inited_model(valid_data)
        outputs = model(valid_data)

        logging.info("torch_nn model infer success".center(60, "="))

        if isinstance(org_outputs, torch.Tensor):
            org_outputs = [org_outputs.data.cpu().numpy()]
        else:
            org_outputs = [tensor.data.cpu().numpy() for tensor in org_outputs]

        if isinstance(outputs, torch.Tensor):
            outputs = [outputs.data.cpu().numpy()]
        else:
            outputs = [tensor.data.cpu().numpy() for tensor in outputs]

        common.output_metric(outputs, org_outputs, 1)
