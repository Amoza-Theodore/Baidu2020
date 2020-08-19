'''
    如影随形的主程序
    1. 摄像头采集图像
    2. 调用模型进行预测, 得出标志物的边界坐标
    3. 根据面积值和中点坐标判断, 更新速度值和角度值
'''
import sys
sys.path.append('../class')

from car import Car
from dlmodel import DLmodel
from mark_status import MarkStatus
from settings import Settings
import functions as func

def follow():
    # 初始化设置选项
    ai_settings = Settings()
    func.clean_img(ai_settings)

    # 初始化小车
    car = Car(ai_settings)

    # 初始化深度学习模型
    dlmodel = DLmodel(ai_settings, follow_flag=True)

    # 创建一个状态类, 用以储存标志物状态
    markstats = MarkStatus()

    # 开始任务的主循环
    while True:
        func.get_image(ai_settings, car, dlmodel)
        func.get_follow_para(ai_settings, dlmodel, markstats)
        func.update_follow_para(ai_settings, car, dlmodel, markstats)

if __name__ == '__main__':
    follow()