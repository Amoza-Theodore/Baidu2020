import numpy as np

class Settings():
    """储存所有跟随任务的设置"""

    def __init__(self):
        """初始化任务设置"""

        # DEBUG选项
        self.DEBUG = False

        # 文件保存路径
        self.angle_model_path = '../model/angle_model/model_infer'
        self.follow_model_path = '../model/follow_model/model_infer'
        self.label_model_path = '../model/label_model/model_infer'
        self.img_save_path = '../predict_img'

        # 小车的设置选项
        self.car_init_speed = 1600
        self.car_init_angle = 1500

        # 车道线识别的设置选项
        self.lower_hsv = np.array([26, 43, 46])
        self.upper_hsv = np.array([34, 255, 255])
        self.angle_formula = lambda angle_prep:int(angle_prep * 4000 - 600)
        self.angle_limit_formula = lambda angle_prep:\
            int(-1792 * angle_prep**2 + 3413 * angle_prep + 176.7 + 2)

        # 标志物检测的设置选项
        self.score_thresold = 0.5
        self.search_time = 2 # 小车在丢失标志物后尝试自动转向搜索的时间
        self.follow_dict = {0:'landmark'}
        self.label_dict = {
            0: 'cone_bucket',
            1: 'limit',
            2: 'limit_end',
            3: 'outer',
            4: 'overtake',
            5: 'red_line',
            6: 'side_walk',
            7: 'back',
            8: 'straight',
            9: 'turn_left'
        }
        #   当标志物的center_y为多大时, 开始检测标志物
        self.dist_control = {
            'side_walk': 70,
            'limit': 70,
            'limit_end': 70,
            'overtake': 30
        }

        # 算法控制的设置选项
        self.sidewalk_pausetime = 1

        # 如影随形的设置选项
        #   角度放大时所用的区间
        self.angle_thre_left = 700
        self.angle_thre_right = 2300
        #   小车在丢失标志物时是否尝试自动转向搜索
        self.remain_search_flag = True
        #   小车在丢失标志物时尝试自动转向搜索所用的角度值
        self.remain_angle_left = 2000
        self.remain_angle_right = 1000
        #   小车在丢失标志物时判断前一次x轴中点坐标偏左或偏右所用的值
        self.centerx_thre_left = 120
        self.centerx_thre_right = 200