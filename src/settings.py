class Settings():
    """储存所有跟随任务的设置"""

    def __init__(self):
        """初始化任务设置"""

        # 模型文件保存路径
        self.angle_model_path = '../model/angle_model/model_infer'
        self.label_model_path = '../model/label_model/model_infer'

        # 小车的设置选项
        self.car_init_speed = 1600
        self.car_init_angle = 1500

        # 标志物检测的设置选项
        self.score_thresold = 0.6