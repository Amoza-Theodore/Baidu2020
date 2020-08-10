'''
    1. 摄像头采集图像
    2. 调用模型进行预测, 得出标志物的中点坐标
    3. 根据中点坐标判断, 更新速度值和角度值
'''

from car import Car
from settings import Settings
import functions as func

def follow():
    # 初始化任务
    ai_settings = Settings()
    func.create_default_folders()
    func.initialize_predictor()

    # 初始化小车
    car = Car(ai_settings)

    # 开始任务的主循环
    while True:
        func.get_images()
        func.get_center_coordinates()
        func.update_vel_and_angle()

if __name__ == '__main__':
    follow()