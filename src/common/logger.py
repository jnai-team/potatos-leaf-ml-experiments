#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ===============================================================================
#
# Copyright (c) 2020 <> All Rights Reserved
#
#
# File: /Users/hain/chatopera/chatopera.xinli/alchemist/app/common/logger.py
# Author: Hai Liang Wang
# Date: 2020-03-21:14:44:42
#
# ===============================================================================

"""
   
"""
import logging
__copyright__ = "Copyright (c) 2020 . All Rights Reserved"
__author__ = "Hai Liang Wang"
__date__ = "2020-03-21:14:44:42"

import os
import sys
import json

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(curdir, os.pardir))

if sys.version_info[0] < 3:
    raise RuntimeError("Must be using Python 3")
else:
    unicode = str

LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
print("LOG_LEVEL %s" % LOG_LEVEL)
logging.basicConfig(level=LOG_LEVEL)

# 日志级别
CRITICAL = 50
FATAL = CRITICAL
ERROR = 40
WARNING = 30
WARN = WARNING
INFO = 20
DEBUG = 10
NOTSET = 0


class Logger(logging.Logger):
    """
    Logger
    """

    def __init__(self, name, level=DEBUG):
        self.name = name
        self.level = level
        logging.Logger.__init__(self, self.name, level=level)
        self.__setStreamHandler__()

    def __setStreamHandler__(self, level=None):
        """
        set stream handler
        :param level:
        :return:
        """
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(levelname)s %(asctime)s %(name)s.%(funcName)s@%(lineno)d: %(message)s')
        stream_handler.setFormatter(formatter)
        if not level:
            stream_handler.setLevel(LOG_LEVEL)
        else:
            stream_handler.setLevel(level)
        self.addHandler(stream_handler)


def pretty(j, indent=4, sort_keys=True):
    """
    get json as pretty string
    :param j:
    :return:
    """
    return json.dumps(j, indent=indent, sort_keys=sort_keys, ensure_ascii=False, default=str)


def LN(x):
    """
    log name 获得模块名字，输出日志短名称
    """
    return x.split(".")[-1]
