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

import numpy as np

def get_image(ai_settings, dlmodel):
    """摄像头获取图像"""
    lower_hsv = np.array([25, 75, 190])
    upper_hsv = np.array([40, 255, 255])

    label_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    dlmodel.img_label = Image.fromarray(label_img)

def get_center_coordinates():
    paddle_data_feeds = dlmodel.deal_tensor(dl.label_image)
    label_outputs = dlmodel.label_predictor.Run(paddle_data_feeds)
    label_outputs = np.array(label_outputs[0], copy=False)
    stats.detect = check_detect(label_outputs)
    if stats.detect:
        stats.labels, stats.scores, boxes = get_img_para(label_outputs)
        stats.center_x, stats.center_y = analyse_box(boxes)

def check_detect():
    """判断是否检测到标志物"""
    if len(label_outputs.shape) > 1:
        scores = label_outputs[:, 1]
        for score in scores:
            # 若 score > 0.6 则表示识别成功
            if score > ai_settings.score_thresold:
                return True
    return False

def get_img_para(label_outputs):
    mask = label_outputs[:, 1] > ai_settings.score_thresold
    labels = label_outputs[mask, 0].astype('int32')
    scores = label_outputs[mask, 1].astype('float32')
    boxes = label_outputs[mask, 2:].astype('float32')

    return labels, scores, boxes

def analyse_box(boxes):
    pass