import time
class MarkStatus():

    def __init__(self):
        # 初始化标志物检测的状态
        self.detect = False
        self.labels = None
        self.scores = None

        # 如影随形的相关内容
        self.follow_center_x = None
        self.last_follow_center_x = None
        self.follow_center_y = None
        self.lose_mark_flag = True
        self.stdtime = time.time()