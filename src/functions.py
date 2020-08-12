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

def get_center_coordinates(ai_settings, dlmodel, markstats):
    """得到中点坐标"""
    paddle_data_feeds = dlmodel.deal_tensor(dlmodel.label_img)
    label_outputs = dlmodel.label_predictor.Run(paddle_data_feeds)
    label_outputs = np.array(label_outputs[0], copy=False)
    markstats.detect = check_detect(ai_settings, label_outputs)
    if markstats.detect:
        labels, scores, markstats.bbox = get_img_para(ai_settings, label_outputs)
        markstats.center_x, markstats.center_y = analyse_box(ai_settings, labels, markstats.bbox)
    else:
        pass
        # save_img(ai_settings, dlmodel)

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

def update_vel_and_angle(ai_settings, car, dlmodel, markstats):
    if markstats.detect:
        car.stop_sign = False
        # 计算角度值并更新 [0, 320] -> [500, 2500]
        car.angle = int(1500 + (160 - markstats.center_x) / 320 * 2000)
        print('car_angle = ' + str(car.angle))
        car.speed = 1600
        car.update()

        # 计算速度值并更新 [0, 240] -> [1500, 1600]
        # speed = 1500 + int(stats.center_y / 240)
        pass
    else:
        if not car.stop_sign:
            remain_search(ai_settings)
        # 未检测到标志物, 小车停止运行
        car.stop()

def remain_search(ai_settings, car):
    """小车在丢失标志物时尝试自动转向搜索"""
    mask = np.array(car.angle_value) < 1500
    car.angle = 500 if sum(mask) > len(mask) / 2 else 2500
    car.update()
    time.sleep(ai_settings.search_time)

def save_img(ai_settings, dlmodel):
    """保存图片"""
    output_path = os.path.join(ai_settings.img_save_path, str(dlmodel.ImgInd) + '.jpg')
    dlmodel.label_img.save(output_path)
    dlmodel.ImgInd += 1

def clean_img(ai_settings):
    """清空 predict_img 文件夹"""
    img_save_path = ai_settings.img_save_path
    if os.path.exists(img_save_path):
        shutil.rmtree(img_save_path)
        os.makedirs(img_save_path)