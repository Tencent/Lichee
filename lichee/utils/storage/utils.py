# -*- coding: utf-8 -*-
from lichee import plugin
from .storage_base import BaseStorage


def get_storage_file(storage_path: str):
    """
    get storage file, return a local file path

    Parameters
    ----------
    storage_path: str
        storage path, a local path, cos path or http, https url

    Returns
    ------
    local_path: str
        local file path
    """
    storage_type_plugin, storage_file_path = analyse_storage(storage_path)
    return storage_type_plugin.get_file(storage_file_path)


def put_storage_file(local_file_path: str, storage_path: str):
    """
    put local file to storage

    Parameters
    ----------
    local_file_path: str
        local file path
    storage_path: str
        storage path, a local path or cos path
    """
    storage_type_plugin, storage_file_path = analyse_storage(storage_path)
    return storage_type_plugin.put_file(local_file_path, storage_file_path)


def analyse_storage(storage_path: str) -> (BaseStorage, str):
    """
    analyse storage type and path

    :param storage_path: str
    :return: storage instance, storage path
    """
    storage_path_arr = storage_path.split("://")
    if len(storage_path_arr) != 2:
        raise Exception("Not supported storage_path: %s" % storage_path)

    storage_type = storage_path_arr[0]
    storage_file_path = storage_path_arr[1]

    storage_type_plugin: BaseStorage = plugin.get_plugin(plugin.PluginType.UTILS_STORAGE, storage_type)
    if storage_type_plugin is None:
        raise Exception("cannot find storage plugin: %s" % storage_path)

    return storage_type_plugin, storage_file_path
