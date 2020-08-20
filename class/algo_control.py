import time

class AlgoControl():
    """储存算法控制的所有相关内容"""
    def __init__(self, ai_settings, car):
        self.ai_settings = ai_settings
        self.car = car

        # 小车控制标志
        self.sidewalk_flag = False
        self.stop_flag = False
        self.limit_flag = False
        self.overtake_flag = False
        self.completed = []

        # 小车运行参数
        self.angle_prep = None


    def overtake(self):
        """弯道超车"""
        car = self.car

        car.turn_left(2150)
        time.sleep(1.3)

        car.upwards()
        time.sleep(0.5)

        car.turn_right(850)
        time.sleep(0.8)

        car.turn_right(1100)
        time.sleep(0.7)