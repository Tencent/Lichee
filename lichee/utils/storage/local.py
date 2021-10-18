# -*- coding: utf-8 -*-
from . import storage_base
from lichee import plugin
import shutil
import os


@plugin.register_plugin(plugin.PluginType.UTILS_STORAGE, "local")
class LocalStorage(storage_base.BaseStorage):
    """
    local storage
    """
    @classmethod
    def get_file(cls, file_path):
        if file_path.startswith("/"):
            return file_path
        else:
            dirname = os.getcwd()
            return os.path.join(dirname, file_path)

    @classmethod
    def put_file(cls, src_file_path: str, dst_file_path: str):
        if not super(LocalStorage, cls).should_put_file():
            return

        dst_file_dir = os.path.dirname(dst_file_path)
        if not os.path.exists(dst_file_dir):
            os.makedirs(dst_file_dir)
        shutil.copy(src_file_path, dst_file_path)
