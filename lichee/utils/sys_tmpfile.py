# -*- coding: utf-8 -*-
import atexit
import os
import shutil
import uuid

GLOBAL_REMOTE_FILES = "./remote/lichee"

GLOBAL_TMP_DIR = "./cache/lichee"
GLOBAL_TMP_LOCK_DIR = "./cache/lichee/lock"


def get_global_remote_dir():
    """get global unique remote dir (will not been cleaned)

    """
    if not os.path.exists(GLOBAL_REMOTE_FILES):
        os.makedirs(GLOBAL_REMOTE_FILES)
    return GLOBAL_REMOTE_FILES


def get_remote_file_path(filename):
    """get remote file path

    """
    global_tmp_dir = get_global_remote_dir()
    remote_file_path = os.path.join(global_tmp_dir, filename)
    return remote_file_path


def get_global_temp_dir():
    """get global unique temp dir

    """
    if not os.path.exists(GLOBAL_TMP_DIR):
        os.makedirs(GLOBAL_TMP_DIR)
    return GLOBAL_TMP_DIR


def get_temp_file_path_once():
    """get empty temp file path which will be cleaned upon exit

    """
    global_tmp_dir = get_global_temp_dir()
    tmp_file_path = os.path.join(global_tmp_dir, uuid.uuid4().hex)
    return tmp_file_path


def get_temp_dir_once():
    """get empty temp dir which will be cleaned upon exit

    """
    global_tmp_dir = get_global_temp_dir()
    tmp_dir = os.path.join(global_tmp_dir, uuid.uuid4().hex)
    os.makedirs(tmp_dir)
    return tmp_dir


def get_global_temp_lock_dir():
    """get lock file dir

    """
    if not os.path.exists(GLOBAL_TMP_LOCK_DIR):
        os.makedirs(GLOBAL_TMP_LOCK_DIR)
    return GLOBAL_TMP_LOCK_DIR


def create_temp_lock_file(filename: str):
    """create lock file

    """
    global_tmp_lock_dir = get_global_temp_lock_dir()
    tmp_lock_file_path = os.path.join(global_tmp_lock_dir, filename + ".lock")
    with open(tmp_lock_file_path, mode="w+"):
        pass
    return tmp_lock_file_path


def exist_temp_lock_file(filename: str):
    """check lock file exist

    """
    global_tmp_lock_dir = get_global_temp_lock_dir()
    tmp_lock_file_path = os.path.join(global_tmp_lock_dir, filename + ".lock")
    return os.path.exists(tmp_lock_file_path)


def clear_tmp_files():
    try:
        shutil.rmtree(GLOBAL_TMP_DIR)
    except:
        pass


atexit.register(clear_tmp_files)
