# -*- coding: UTF-8 -*-
#!/usr/bin/env python

import matplotlib.pyplot as plt
import re
from mpl_toolkits.axes_grid1 import host_subplot

# read the log file
fp = open('faster_rcnn_end2end_ResNet-50-model_.txt', 'r')

train_iterations = []
lr_array = []


for ln in fp:
  # get train_iterations and train_loss
  if '] Iteration ' in ln and 'lr = ' in ln:
    arr = re.findall(r'ion \b\d+\b,',ln)
    print arr[0].strip(',')[2:]#n 2990
    train_iterations.append(int(arr[0].strip(',')[4:]))
    lr_array.append(float(ln.strip().split(' = ')[-1]))

fp.close()

host = host_subplot(111)
plt.subplots_adjust(right=0.8) # ajust the right boundary of the plot window
# par1 = host.twinx()#设置双坐标

# set labels
host.set_xlabel("iterations")
host.set_ylabel("lr")

# plot curves
p1, = host.plot(train_iterations, lr_array, label="training lr")

# set location of the legend,
# 1->rightup corner, 2->leftup corner, 3->leftdown corner
# 4->rightdown corner, 5->rightmid ...
host.legend(loc=1)

# set label color
host.axis["left"].label.set_color(p1.get_color())

# set the range of x axis of host and y axis of par1
host.set_xlim([train_iterations[0], train_iterations[-1]+5])
host.set_ylim([lr_array[-1]*0.9, lr_array[0]*1.1])

plt.draw()
plt.show()
