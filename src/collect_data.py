'''
    1. 获取手柄信息
    2. 调整小车运行的参数
    3. 储存图片和数值
'''

import os
import v4l2capture
import select
from ctypes import *
import struct, array
from fcntl import ioctl
import cv2
import numpy as np
import time
from sys import argv
import multiprocessing
import time
import getopt

from settings import Settings
import functions as func

def collect_data():
    # 初始化设置选项
    ai_settings = Settings()

    # 初始化小车
    car = Car(ai_settings)

    # 开始任务的主循环
    while True:
        func.check_events()
        func.update_collect_para()
        func.restore_data()

if __name__ == '__main__':
    collect_data()