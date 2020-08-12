'''
    1. 摄像头采集图像
    2. 调用模型进行预测, 得出标志物的中点坐标
    3. 根据中点坐标判断, 更新速度值和角度值
'''

import sys

from car import Car
from deep_learning_model import DLmodel
from markstatus import MarkStatus
from settings import Settings
import pid as PID
import functions as func

def follow():
    # 初始化设置选项
    ai_settings = Settings()

    # 初始化小车
    car = Car(ai_settings)

    # 初始化深度学习模型
    dlmodel = DLmodel(ai_settings)

    # 创建一个状态类, 用以储存状态
    markstats = MarkStatus()

    # 开始任务的主循环
    while True:
        func.get_image(ai_settings, car, dlmodel)
        func.get_center_coordinates(ai_settings, dlmodel, markstats)
        func.update_vel_and_angle(ai_settings, car, dlmodel, markstats)

if __name__ == '__main__':
    follow()