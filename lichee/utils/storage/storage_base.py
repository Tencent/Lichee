# -*- coding: utf-8 -*-
from lichee.utils import common


class BaseStorage:
    @classmethod
    def get_file(cls, file_path: str):
        pass

    @classmethod
    def put_file(cls, src_file_path: str, dst_file_path: str):
        pass

    @classmethod
    def should_put_file(cls):
        # exec before put_file
        # only put_file by rank=0 in DDP mode
        return common.is_rank_0()
