# -*- coding: utf-8 -*-
import logging
from collections import defaultdict

import torch

from lichee import config
from lichee import plugin
from lichee.model import model_base
from lichee.utils import storage


class Unit:
    def __init__(self, idx, name, inputs, outputs):
        self.idx = idx
        self.name = name
        self.inputs = inputs
        self.outputs = outputs


@plugin.register_plugin(plugin.PluginType.MODEL, "model_standard")
class ModelStandard(model_base.BaseModel):
    def __init__(self, cfg, requires_grad=False):
        '''
        :param cfg: model config
        :param requires_grad: whether require grad or not TODO: fix requires_grad
        '''
        super(ModelStandard, self).__init__(cfg)
        self.task = None
        self.model_index = {}
        self.label_keys = []
        self.target_input_names = []
        self.topo_units = []
        # get model inputs
        self.model_inputs = config.get_model_inputs()
        # build representations
        self.build_representations()
        # build cls for target
        self.build_cls()
        self.graph_units = None
        # build graph units for topo
        self.build_graph_units()
        # build the topo units
        self.build_topo_units()

    def build_representations(self):
        # 创建representation和target
        for representation_cfg in self.cfg.MODEL.REPRESENTATION:
            representation_cls = plugin.get_plugin(plugin.PluginType.REPRESENTATION, representation_cfg["TYPE"])
            # 预训练模型初始化
            if "PRETRAINED" in representation_cfg and representation_cfg["PRETRAINED"]:
                # 获取模型路径
                model_path = storage.get_storage_file(representation_cfg["MODEL_PATH"])
                logging.info('recover {} pretrain parameters from: {}'.format(representation_cfg["NAME"], model_path))
                representation_cls = representation_cls(representation_cfg)
                representation_model = representation_cls.load_pretrained_model(
                    representation_cfg, model_path)
            else:
                representation_model = representation_cls(representation_cfg)
            logging.info(
                'initialize {} with parameters: {}'.format(representation_cfg["NAME"],
                                                           len(list(representation_model.parameters()))))
            # fine-tuning设置，默认不开启
            if "FINE_TUNING" in representation_cfg and representation_cfg["FINE_TUNING"]:
                requires_grad = representation_cfg["FINE_TUNING"]
                for param in representation_model.parameters():
                    param.requires_grad = requires_grad

            # 记录索引
            self.model_index[representation_cfg["NAME"]] = len(self.representation_model_arr)
            self.representation_model_arr.append(representation_model)

    def build_cls(self):
        task_cls = plugin.get_plugin(plugin.PluginType.TASK, self.cfg.MODEL.TASK.NAME)
        self.task = task_cls(self.cfg.MODEL.TASK)

    def build_graph_units(self):
        # topo units prepare
        units = []
        output_record = defaultdict(list)
        for graph_cfg in self.cfg.MODEL.GRAPH:
            name = graph_cfg["NAME"]
            if "LABELS" in graph_cfg:
                self.target_input_names = [input.strip() for input in graph_cfg["INPUTS"]]
                self.label_keys.extend(graph_cfg["LABELS"].split(','))
                continue
            elif name not in self.model_index:
                raise Exception("ERROR: model %s not exist in representation config!" % name)
            idx = self.model_index[name]
            inputs = [input.strip() for input in graph_cfg["INPUTS"]]
            outputs = [output.strip() for output in graph_cfg["OUTPUTS"]]
            for output in outputs:
                output_record[output].append(name)
                # check invalid model outputs. The output key should not be in model input.
                if output in self.model_inputs:
                    raise Exception(
                        "ERROR: representation {} output key {} can't be in model inputs: {}".format(name, output,
                                                                                                     self.model_inputs))
            unit = Unit(idx, name, inputs, outputs)
            units.append(unit)
        invalid_ouputs = [[output, name_list] for output, name_list in output_record.items() if len(name_list) != 1]
        if invalid_ouputs:
            # output key should not duplicated!
            raise Exception(
                'ERROR: Invalid duplicated output key {} for representations {}'.format([x[0] for x in invalid_ouputs],
                                                                                        [x[1] for x in invalid_ouputs]))
        self.graph_units = units

    def build_topo_units(self):
        # topo units generate
        units = self.graph_units
        model_inputs_set = set(self.model_inputs)
        while True:
            num_topo_units_before = len(self.topo_units)
            for unit in units:
                unit_inputs_set = set(unit.inputs)
                if unit_inputs_set.issubset(model_inputs_set):
                    self.topo_units.append(unit)
                    for output in unit.outputs:
                        model_inputs_set.add(output)
            units = [unit for unit in units if unit not in self.topo_units]
            if not units:
                break
            num_topo_units_after = len(self.topo_units)
            if num_topo_units_before == num_topo_units_after:  # True if the topo process is stucked
                raise Exception('''
                    ERROR: success to build topo representations for {}
                    with inputs {}
                    but failed to build for {} for inadequate inputs. 
                '''.format([x.name for x in self.topo_units], model_inputs_set,
                           [{'representation': x.name, 'inputs': x.inputs} for x in units]))
        # check target input names
        for target_input in self.target_input_names:
            if target_input not in model_inputs_set:
                raise Exception(
                    "ERROR: the required input {} not in input features {}".format(target_input, model_inputs_set))

    def forward(self, inputs):
        '''
        :param inputs: input batch data with {key1: batch_feature1, key2:batch_feature2}
        :return: classification result
        '''
        inputs = inputs.copy()

        # 按照拓扑排序的顺序执行
        for unit in self.topo_units:
            model = self.representation_model_arr[unit.idx]
            model_inputs = [inputs[key] for key in unit.inputs]
            model_outputs = model(*model_inputs)
            # update feature map for the following process
            if isinstance(model_outputs, torch.Tensor) or len(unit.outputs) == 1:  # single output
                assert len(unit.outputs) == 1
                inputs[unit.outputs[0]] = model_outputs
            else:  # multiple output
                assert len(model_outputs) == len(unit.outputs)  # output feature and key has sim size
                inputs.update(dict(zip(unit.outputs, model_outputs)))
        # generate target inputs
        target_inputs = []
        for key in self.target_input_names:
            target_inputs.append(inputs[key] if key in inputs else None)

        # generate label inputs
        label_inputs = {}
        for label_key in self.label_keys:
            if label_key not in inputs:
                continue
            label_inputs[label_key] = inputs[label_key]
        # return with task outputs
        return self.task(*target_inputs, label_inputs if len(label_inputs) > 0 else None)

    def independent_lr_parameters(self):
        params = []
        for representation in self.representation_model_arr:
            params += representation.independent_lr_parameters()
        return params
