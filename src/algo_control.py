class AlgoControl():
    """储存算法控制的所有相关内容"""
    def __init__(self):
        # 小车控制标志
        self.stop_flag = False
        self.limit_flag = False

        # 小车运行参数
        self.angle_prep = None
