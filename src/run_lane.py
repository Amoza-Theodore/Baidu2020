"""
    车道线识别的主程序
    1. 摄像头采集图像
    2. 调用模型进行预测, 得出当前图片所应输出的角度值
    3. 根据角度放大公式进行放大处理, 并更新小车的角度值
"""

from car import Car
from dlmodel import DLmodel
from settings import Settings
import functions as func

def run_line():
    # 初始化设置选项
    ai_settings = Settings()
    func.clean_img(ai_settings)

    # 初始化小车
    car = Car(ai_settings)

    # 初始化深度学习模型
    dlmodel = DLmodel(ai_settings)

    # 开始任务的主循环
    while True:
        func.get_image(ai_settings, car, dlmodel)
        angle_prep = func.predict_angle()
        angle = ai_settings.angle_formula(angle_prep)
        car.update(angle=angle)

if __name__ == '__main__':
    run_line()