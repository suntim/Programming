#_*_coding:utf-8_*_
#!/usr/bin/env python
'__author__' == 'Alex_XT'
import matplotlib.pyplot as plt
import numpy as np

def smoothL1(x):
    if abs(x)<1:
        return 0.5*x**2;
    else:
        return abs(x)-0.5




if __name__ == '__main__':
    x = np.linspace(-20,20,101)*1.0/10
    # print x
    y = np.array([smoothL1(i) for i in x])
    # print y
    fig1 =plt.figure('smoothL1')
    plt.plot(x,y,'k-')

    inds = np.where(abs(x)<1)[0]
    x1 = x[inds]
    y1 = y[inds]
    # print y1
    plt.plot(x1, y1, 'k-o')

    plt.xlabel('x')
    plt.ylabel('smoothL1')
    plt.show()




