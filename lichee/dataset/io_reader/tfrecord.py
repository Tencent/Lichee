# -*- coding: utf-8 -*-
from lichee import plugin
from .io_reader_base import TFRecordReader


@plugin.register_plugin(plugin.PluginType.DATA_IO_READER, "tfrecord")
class TFRecordReaderPlugin(TFRecordReader):
    """
    tfrecord reader implementation
    """
    pass
