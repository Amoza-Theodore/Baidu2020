import os, shutil
import v4l2capture
from ctypes import *
import struct, array
from fcntl import ioctl
import cv2
import numpy as np
import time
from sys import argv
import getopt
import sys, select, termios, tty
import threading
import paddlemobile as pm
from paddlelite import *
import codecs
import multiprocessing
import math
import functools
from PIL import Image
from PIL import ImageFile
from PIL import ImageDraw
ImageFile.LOAD_TRUNCATED_IMAGES = True

class DLmodel():
    """储存deep_learning_model的所有相关内容"""
    def __init__(self, ai_settings, follow_flag=False):
        self.ai_settings = ai_settings

        # 初始化角度预测器
        valid_places = (
            Place(TargetType.kFPGA, PrecisionType.kFP16, DataLayoutType.kNHWC),
            Place(TargetType.kHost, PrecisionType.kFloat),
            Place(TargetType.kARM, PrecisionType.kFloat),
        );
        config = CxxConfig();
        model_dir = ai_settings.angle_model_path;
        config.set_model_file(model_dir + "/model");
        config.set_param_file(model_dir + "/params");
        config.set_valid_places(valid_places);
        self.angle_predictor = CreatePaddlePredictor(config);

        # 初始化标志预测器
        if follow_flag:
            label_model_path = ai_settings.follow_model_path
        elif not follow_flag:
            label_model_path = ai_settings.label_model_path
        model_dir = label_model_path
        pm_config = pm.PaddleMobileConfig()
        pm_config.precision = pm.PaddleMobileConfig.Precision.FP32
        pm_config.device = pm.PaddleMobileConfig.Device.kFPGA
        pm_config.model_dir = model_dir
        pm_config.thread_num = 4
        self.label_predictor = pm.CreatePaddlePredictor(pm_config)

        # 标志物相关变量
        self.label_img = None
        self.angle_img = None
        self.ImgInd = 0

    def deal_tensor(self):
        tensor_img = self.label_img.resize((256, 256), Image.BILINEAR)
        if tensor_img.mode != 'RGB':
            tensor_img = tensor_img.convert('RGB')
        tensor_img = np.array(tensor_img).astype('float32').transpose((2, 0, 1))
        tensor_img -= 127.5
        tensor_img *= 0.007843
        tensor_img = tensor_img[np.newaxis, :]
        tensor = pm.PaddleTensor()
        tensor.dtype = pm.PaddleDType.FLOAT32
        tensor.shape = (1, 3, 256, 256)
        tensor.data = pm.PaddleBuf(tensor_img)
        paddle_data_feeds = [tensor]
        return paddle_data_feeds