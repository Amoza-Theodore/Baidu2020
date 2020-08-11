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

def get_image(ai_settings, car, dlmodel):
    """摄像头获取图像"""
    lower_hsv = np.array([25, 75, 190])
    upper_hsv = np.array([40, 255, 255])

    select.select((car.video,), (), ())
    image_data = car.video.read_and_queue()
    frame = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
    label_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    dlmodel.label_img = Image.fromarray(label_img)

def get_center_coordinates(ai_settings, dlmodel, stats):
    paddle_data_feeds = dlmodel.deal_tensor(dlmodel.label_img)
    label_outputs = dlmodel.label_predictor.Run(paddle_data_feeds)
    label_outputs = np.array(label_outputs[0], copy=False)
    stats.detect = check_detect(ai_settings, label_outputs)
    if stats.detect:
        print("detect successfully!")
        labels, scores, boxes = get_img_para(ai_settings, label_outputs)
        stats.center_x, stats.center_y = analyse_box(ai_settings, labels, boxes)
    else:
        print("not detect.")

def check_detect(ai_settings, label_outputs):
    """判断是否检测到标志物"""
    if len(label_outputs.shape) > 1:
        scores = label_outputs[:, 1]
        for score in scores:
            # 若 score > 0.6 则表示识别成功
            if score > ai_settings.score_thresold:
                return True
    return False

def get_img_para(ai_settings, label_outputs):
    """处理 label_outputs, 得到 labels, scores, boxes"""
    mask = label_outputs[:, 1] > ai_settings.score_thresold
    labels = label_outputs[mask, 0].astype('int32')
    scores = label_outputs[mask, 1].astype('float32')
    boxes = label_outputs[mask, 2:].astype('float32')

    return labels, scores, boxes

def analyse_box(ai_settings, labels, boxes):
    """处理 boxes, 得到 center_x, center_y"""
    for label_idx, box in zip(labels, boxes):
        if ai_settings.label_dict[label_idx] == 'landmark':
            xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
            xmin, xmax = (int(x / 608 * 320) for x in [xmin, xmax])
            ymin, ymax = (int(y / 608 * 240) for y in [ymin, ymax])
            center_x = int((xmin + xmax) / 2)
            center_y = int((ymin + ymax) / 2)
            return center_x, center_y

def update_vel_and_angle(ai_settings, car, stats):
    if stats.detect:
        # 计算角度值并更新 [0, 320] -> [900, 2100]
        angle = 900 + int(stats.center_x / 320 * 1200)
        car.angle = angle
        car.update()

        # 计算速度值并更新 [0, 240] -> [1500, 1600]
        # speed = 1500 + int(stats.center_y / 240)
        pass
    else:
        time.sleep(0.5)
        # 未检测到标志物, 小车停止运行
        car.stop()