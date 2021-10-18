# -*- coding: utf-8 -*-
from lichee import plugin
from . import loader_base


@plugin.register_plugin(plugin.PluginType.UTILS_MODEL_LOADER, "onnx")
class ONNXLoader(loader_base.BaseLoader):
    """provides ability to load model from onnx format

    Attributes
    ----------
    ort_sess: ort.InferenceSession
        loaded model instance

    """

    def __init__(self, model_path):
        super(ONNXLoader, self).__init__(model_path)
        import onnxruntime as ort
        self.ort_sess = ort.InferenceSession(model_path)

    def forward(self, inputs):
        for key in inputs:
            inputs[key] = inputs[key].cpu().numpy()
        return self.ort_sess.run(None, inputs)
