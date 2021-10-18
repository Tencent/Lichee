# -*- coding: utf-8 -*-
class BaseConvertor:
    """ConvertorBase Class. Provide two methods for usage.

    Attributes
    ----------
    save_model: save model with specific type

    check_and_infer: check model and do inference for accuracy check

    """
    @classmethod
    def convert(cls, inited_model, trace_inputs, sample_inputs, export_model_path):
        cls.save_model(inited_model, trace_inputs, export_model_path)
        cls.check_and_infer(inited_model, sample_inputs, export_model_path)

    @classmethod
    def save_model(cls, inited_model, trace_inputs, export_model_path):
        pass

    @classmethod
    def check_and_infer(cls, inited_model, valid_data, export_model_path):
        pass
