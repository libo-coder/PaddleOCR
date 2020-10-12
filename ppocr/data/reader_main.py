# coding=utf-8
"""
@author: libo
"""
import os
import random
import numpy as np

import paddle
from ppocr.utils.utility import create_module
from copy import deepcopy

from .rec.img_tools import process_image
import cv2

import sys
import signal


# handle terminate reader process, do not print stack frame
def _reader_quit(signum, frame):
    print("Reader process exit.")
    sys.exit()


def _term_group(sig_num, frame):
    print('pid {} terminated, terminate group {}...'.format(os.getpid(), os.getpgrp()))
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)


signal.signal(signal.SIGTERM, _reader_quit)
signal.signal(signal.SIGINT, _term_group)


def reader_main(config=None, mode=None):
    """Create a reader for trainning

    Args:
        config: arguments
        mode(str): train or val or test

    Returns:
        train reader
    """
    assert mode in ["train", "eval", "test"], "Nonsupport mode:{}".format(mode)
    global_params = config['Global']
    if mode == "train":
        params = deepcopy(config['TrainReader'])
    elif mode == "eval":
        params = deepcopy(config['EvalReader'])
    else:
        params = deepcopy(config['TestReader'])
    params['mode'] = mode
    params.update(global_params)    # 将引入的两个 yml 配置文件的参数进行融合
    reader_function = params['reader_function']     # ppocr.data.det.dataset_traversal, TrainReader
    function = create_module(reader_function)(params)
    if mode == "train":
        if sys.platform == "win32":
            return function(0)
        readers = []
        num_workers = params['num_workers']
        for process_id in range(num_workers):
            readers.append(function(process_id))
        return paddle.reader.multiprocess_reader(readers, False)
    else:
        return function(mode)
