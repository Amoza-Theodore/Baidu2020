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

class Car():

    def __init__(self, ai_settings):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self.ai_settings = ai_settings

        # 初始化摄像头
        camera = "/dev/video2"
        video = v4l2capture.Video_device(camera)
        video.set_format(424, 240, fourcc='MJPG')
        video.create_buffers(1)
        video.queue_all_buffers()
        video.start()
        self.video = video

        # 初始化底层 arduino
        path = os.path.split(os.path.realpath(__file__))[0] + "/.."
        lib_path = path + "/lib" + "/libart_driver.so"
        so = cdll.LoadLibrary
        self.lib = so(lib_path)
        car = "/dev/ttyUSB0"
        self.lib.art_racecar_init(38400, car.encode("utf-8"))

        # 初始化速度值和角度值
        self.speed = ai_settings.car_init_speed
        self.angle = ai_settings.car_init_angle

    def go_straight(self):
        self.angle = 1500
        self.update()

    def stop(self):
        self.speed = 1500
        self.update()

    def update(self):
        # self.lib.send_cmd(self.speed, self.angle)
        pass