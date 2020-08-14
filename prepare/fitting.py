from scipy.interpolate import KroghInterpolator
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi
from scipy.interpolate import interp1d
from scipy.interpolate import BarycentricInterpolator
import numpy.polynomial.chebyshev as chebyshev
import numpy.linalg as linalg
# 数据准备

print(X)
print(Y)
z1 = np.polyfit(X, Y, 6) # 用5次多项式拟合
p1 = np.poly1d(z1)
print('多项式表达式：',p1) # 在屏幕上打印拟合多项式
Yvals=p1(X)
print(Y-Yvals)
plot1=plt.plot(X, Y, '*',label='original values')
plot2=plt.plot(X, Yvals, 'r',label='polyfit values')
plt.legend(loc=4)
plt.show()
print('\n\n\n\n\n\n\n\n')

angle = 0.772

a = int(28590*angle*angle*angle*angle*angle*angle-95330*angle*angle*angle*angle*angle+114200*angle*angle*angle*angle-63310*angle*angle*angle+19020*angle*angle-1969*angle+932.8)
print(a)
print(int(p1(angle)))