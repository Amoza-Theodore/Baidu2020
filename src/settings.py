class Settings():
    """储存所有跟随任务的设置"""

    def __init__(self):
        """初始化任务设置"""

        # 文件保存路径
        self.label_model_path = '../model/label_model/model_infer'
        self.img_save_path = '../predict_img'

        # 小车的设置选项
        self.car_init_speed = 1600
        self.car_init_angle = 1500

        # 标志物检测的设置选项
        self.score_thresold = 0.6
        self.search_time = 2 # 小车在丢失标志物后尝试自动转向搜索的时间
        self.label_dict = {
            0: 'landmark'
        }

        # 手柄信息
