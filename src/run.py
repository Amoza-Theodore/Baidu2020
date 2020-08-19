"""
    智能交通的主程序: (包括车道线识别和标志物检测)
    1. 摄像头采集图像
    2. 调用角度值模型进行预测, 得出当前图片所应输出的角度值
    3. 调用标志物模型进行预测, 得出当前图片有关标志物的信息
    4. 对不同标志物分别进行算法层面的控制
"""
import sys
sys.path.append('../class')

from car import Car
from dlmodel import DLmodel
from mark_status import MarkStatus
from algo_control import AlgoControl
from settings import Settings
import functions as func

def run():
    # 初始化设置选项
    ai_settings = Settings()
    func.clean_img(ai_settings)

    # 初始化小车
    car = Car(ai_settings)

    # 初始化深度学习模型
    dlmodel = DLmodel(ai_settings)

    # 初始化算法控制系统
    algocontrol = AlgoControl()

    # 创建一个状态类, 用以储存标志物状态
    markstats = MarkStatus()

    # 开始任务的主循环
    while True:
        func.get_image(ai_settings, car, dlmodel)
        algocontrol.angle_prep = func.predict_angle(ai_settings, dlmodel)
        car.angle = ai_settings.angle_formula(algocontrol.angle_prep)
        func.get_label_para(ai_settings, dlmodel, markstats)
        func.algorithmal_control(ai_settings, car, dlmodel, markstats, algocontrol)

if __name__ == '__main__':
    run()