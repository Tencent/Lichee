# -*- coding: utf-8 -*-

"""
plugin interface
"""
from enum import IntEnum
from enum import unique
from typing import Dict


@unique
class PluginType(IntEnum):
    """Plugin类型"""
    TRAINER = 1

    EVALUATOR = 2

    PREDICTOR = 3

    MODEL = 4

    REPRESENTATION = 5

    TASK = 6

    TASK_OUTPUT = 7

    DATA_LOADER = 8

    FIELD_PARSER = 9

    DATA_IO_READER = 10

    SAMPLER = 11

    MODULE_LAYER = 20

    MODULE_LOSS = 21

    MODULE_OP = 22

    MODULE_OPTIMIZER = 23

    MODULE_SCHEDULER = 24

    MODULE_METRICS = 25

    UTILS_CONVERTOR = 40

    UTILS_MODEL_LOADER = 41

    UTILS_STORAGE = 42


class PluginManager:
    """Plugin管理器"""

    def __init__(self):
        self.plugin_container: Dict[str:Dict[str:object]] = {}

    def register(self, plugin_type: PluginType, plugin_name: str, plugin_cls):
        """
        注册plugin

        :param plugin_type: plugin类型
        :param plugin_name: plugin名
        :param plugin_cls: plugin类或函数
        :return:
        """
        if plugin_type not in self.plugin_container:
            self.plugin_container[plugin_type] = {}

        self.plugin_container[plugin_type][plugin_name] = plugin_cls

    def get(self, plugin_type: PluginType, plugin_name: str):
        """
        根据plugin类型及plugin名获取plugin类

        :param plugin_type: plugin类型
        :param plugin_name: plugin名
        :return:
        """
        if plugin_type in self.plugin_container and plugin_name in self.plugin_container[plugin_type]:
            return self.plugin_container[plugin_type][plugin_name]
        else:
            return None


DefaultPluginManager = PluginManager()


def register_plugin(plugin_type: PluginType, plugin_name: str):
    """
    注册plugin的装饰器
    """

    def decorator(cls):
        DefaultPluginManager.register(plugin_type, plugin_name, cls)
        return cls

    return decorator


def get_plugin(plugin_type: PluginType, plugin_name: str):
    """
    获取plugin，注意：此方法必须在setup之后调用

    :param plugin_type: plugin类型
    :param plugin_name: plugin名
    :return:
    """
    return DefaultPluginManager.get(plugin_type, plugin_name)
