# -*- coding: utf-8 -*-
import logging

import torch

from lichee import config
from lichee.utils import common
from lichee import plugin
from lichee.utils.convertor import convertor_base


@plugin.register_plugin(plugin.PluginType.UTILS_CONVERTOR, "onnx")
class ONNXConvertor(convertor_base.BaseConvertor):
    @classmethod
    def save_model(cls, inited_model, trace_inputs, export_model_path):
        if isinstance(inited_model, torch.nn.DataParallel):
            inited_model = inited_model.module

        inited_model.eval()

        cfg = config.get_cfg()
        input_keys = config.get_model_inputs()

        if cfg.RUNTIME.EXPORT.DYNAMIC:
            dynamic_axes = {}
            for key in input_keys:
                dynamic_axes[key] = {}
                size = trace_inputs[key].size()
                for i in range(len(size)):
                    dynamic_axes[key][i] = "dim_" + str(i)

            torch.onnx.export(inited_model, trace_inputs, export_model_path, verbose=True,
                              input_names=input_keys, output_names=["output"], dynamic_axes=dynamic_axes)
        else:
            torch.onnx.export(inited_model, trace_inputs, export_model_path, verbose=True,
                              input_names=input_keys, output_names=["output"])

    @classmethod
    def check_and_infer(cls, inited_model, valid_data, export_model_path):
        # check
        import onnx
        # Load the ONNX model
        model = onnx.load(export_model_path)

        # Check that the IR is well formed
        logging.info("onn model check start".center(60, "="))
        onnx.checker.check_model(model)
        logging.info("onn model check pass".center(60, "="))

        # infer
        org_outputs = inited_model(valid_data)

        logging.info("onn model infer pass".center(60, "="))
        import onnxruntime as ort
        for key in valid_data:
            valid_data[key] = valid_data[key].cpu().numpy()
        ort_sess = ort.InferenceSession(export_model_path)
        outputs = ort_sess.run(None, valid_data)[0]
        logging.info("onn model infer success".center(60, "="))

        # compare result
        logging.info("org outputs: %s", [output.cpu() for output in org_outputs])
        logging.info("onnx outputs: %s", outputs)

        common.output_metric([outputs], [org_output.cpu().detach().numpy() for org_output in org_outputs], 1)
