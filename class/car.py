import os, time
import v4l2capture
from ctypes import *
import tty
from paddlelite import *
from PIL import Image
from PIL import ImageFile

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

        # 初始化底层arduino
        path = os.path.split(os.path.realpath(__file__))[0] + "/.."
        lib_path = path + "/lib" + "/libart_driver.so"
        so = cdll.LoadLibrary
        self.lib = so(lib_path)
        car = "/dev/ttyUSB0"
        self.lib.art_racecar_init(38400, car.encode("utf-8"))

        # 初始化速度值和角度值
        self.speed = ai_settings.car_init_speed
        self.angle = ai_settings.car_init_angle
        self.angle_value = []

    def upwards(self, speed=1600):
        """小车直行"""
        self.speed = speed
        self.angle = 1500
        self.update()

    def backwards(self, speed=1400):
        """小车后退"""
        if speed >= 1450: raise Exception("后退速度值必须小于1450!")
        self.speed = speed
        self.angle = 1500
        self.update()

    def turn_left(self, angle=2100):
        """小车左转"""
        if angle <= 1500: raise Exception("左转角度值必须大于1500!")
        self.angle = angle
        self.update()

    def turn_right(self, angle=900):
        """小车右转"""
        if angle >= 1500: raise Exception("右转角度值必须小于1500!")
        self.angle = angle
        self.update()

    def stop(self):
        """小车停止运行"""
        self.speed = self.angle = 1500
        self.update()

    def jerk(self):
        """小车急停"""
        self.backwards(700)
        time.sleep(0.1)
        self.stop()

    def update(self, speed=None, angle=None):
        """更新小车的速度值和角度值"""
        if speed is None: speed = self.speed
        if angle is None: angle = self.angle

        # 若开启DEBUG, 则不会运行, 不会显示串口发送的命令
        if not self.ai_settings.DEBUG:
            self.lib.send_cmd(speed, angle)