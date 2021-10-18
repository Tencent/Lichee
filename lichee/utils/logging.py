# -*- coding: utf-8 -*-
import logging

from lichee.utils import common


# only logging error by rank=0 with DDP mode
def error(msg, *args, **kwargs):
    if common.is_rank_0():
        logging.warning(msg, *args, **kwargs)


# only logging warning by rank=0 with DDP mode
def warning(msg, *args, **kwargs):
    if common.is_rank_0():
        logging.warning(msg, *args, **kwargs)


# only logging info by rank=0 with DDP mode
def info(msg, *args, **kwargs):
    if common.is_rank_0():
        logging.info(msg, *args, **kwargs)


# only logging debug by rank=0 with DDP mode
def debug(msg, *args, **kwargs):
    if common.is_rank_0():
        logging.debug(msg, *args, **kwargs)
