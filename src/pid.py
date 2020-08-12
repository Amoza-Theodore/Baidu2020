class IncrementalPID:
    """增量式 PID 系统"""
    def __init__(self, P, I, D):
        self.Kp = P
        self.Ki = I
        self.Kd = D
 
        self.PIDOutput = 0.0             # PID控制器输出
        self.SystemOutput = 0.0          # 系统输出值
        self.LastSystemOutput = 0.0      # 上次系统输出值
 
        self.Error = 0.0                 # 输出值与输入值的偏差
        self.LastError = 0.0
        self.LastLastError = 0.0
 
    # 设置PID控制器参数
    def SetStepSignal(self,StepSignal):
        self.Error = StepSignal - self.SystemOutput
        IncrementValue = self.Kp * (self.Error - self.LastError) + self.Ki * self.Error + self.Kd * (self.Error - 2 * self.LastError + self.LastLastError)

        self.PIDOutput += IncrementValue
        self.LastLastError = self.LastError
        self.LastError = self.Error

    # 设置一阶惯性环节系统, 其中InertiaTime为惯性时间常数
    def SetInertiaTime(self,InertiaTime,SampleTime):
        self.SystemOutput = (InertiaTime * self.LastSystemOutput + SampleTime * self.PIDOutput) / (SampleTime + InertiaTime)

        self.LastSystemOutput = self.SystemOutput

class PositionalPID:
    """位置式 PID 系统"""
    def __init__(self, P, I, D):
        self.Kp = P
        self.Ki = I
        self.Kd = D
 
        self.SystemOutput = 0.0
        self.ResultValueBack = 0.0
        self.PidOutput = 0.0
        self.PIDErrADD = 0.0
        self.ErrBack = 0.0
    
    # 设置 PID 控制器参数
    def SetStepSignal(self, StepSignal):
        Err = StepSignal - self.SystemOutput
        KpWork = self.Kp * Err
        KiWork = self.Ki * self.PIDErrADD
        KdWork = self.Kd * (Err - self.ErrBack)
        self.PidOutput = KpWork + KiWork + KdWork
        self.PIDErrADD += Err
        self.ErrBack = Err

    # 设置一阶惯性环节系统, 其中InertiaTime为惯性时间常数
    def SetInertiaTime(self, InertiaTime, SampleTime):
       self.SystemOutput = (InertiaTime * self.ResultValueBack + \
           SampleTime * self.PidOutput) / (SampleTime + InertiaTime)
       self.ResultValueBack = self.SystemOutput
       