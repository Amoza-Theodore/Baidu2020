
class MarkStatus():

    def __init__(self):
        # 初始化标志物检测的状态
        self.detect = False
        self.labels = None
        self.scores = None
        self.center_x = None
        self.center_y = None