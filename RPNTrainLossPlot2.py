# -*- coding: UTF-8 -*-
#!/usr/bin/env python


import matplotlib.pyplot as plt
import re
import numpy as np
from mpl_toolkits.axes_grid1 import host_subplot

# read the log file
# fp = open('/home/alex/Documents/py-R-FCN/experiments/logs/faster_rcnn_end2end_VGG16_.txt.2018-03-05_21-45-13', 'r')
fp = open('/home/alex/Documents/py-R-FCN/experiments/logs/faster_rcnn_end2end_VGG16_.txt.2018-03-05_21-45-13', 'r')

train_iterations = []
train_loss = []
loss_bbox = []
loss_cls = []
rpn_cls_loss = []
rpn_loss_bbox = []

for ln in fp:
    # get train_iterations and train_loss
    if '] Iteration ' in ln and 'loss = ' in ln:
        arr = re.findall(r'ion \b\d+\b,', ln)
        train_iterations.append(int(arr[0].strip(',')[4:]))
        train_loss.append(float(ln.strip().split(' = ')[-1]))
    if 'Train net output ' in ln and ': loss_bbox = ' in ln:
        loss_bbox.append(float(ln.strip().split(' = ')[-1].split(' ')[0]))
    if 'Train net output ' in ln and ': loss_cls = ' in ln:
        loss_cls.append(float(ln.strip().split(' = ')[-1].split(' ')[0]))
    if 'Train net output ' in ln and ': rpn_cls_loss = ' in ln:
        rpn_cls_loss.append(float(ln.strip().split(' = ')[-1].split(' ')[0]))
    if 'Train net output ' in ln and ': rpn_loss_bbox = ' in ln:
        rpn_loss_bbox.append(float(ln.strip().split(' = ')[-1].split(' ')[0]))


fp.close()

host = host_subplot(111)
plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
# par1 = host.twinx()
# set labels
host.set_xlabel("iterations")
host.set_ylabel("loss")

# plot curves
p1, = host.plot(train_iterations, train_loss, 'b-',label="train loss")

# set location of the legend,
# 1->rightup corner, 2->leftup corner, 3->leftdown corner
# 4->rightdown corner, 5->rightmid ...
host.legend(loc=1)

# set label color
host.axis["left"].label.set_color(p1.get_color())
host.set_xlim([train_iterations[0], train_iterations[-1]+5])
host.set_ylim([train_loss[-1], train_loss[0]])

fig2 = plt.figure("fig2_loss_bbox")
print "train_iterations.shape = {} , loss_bbox.shape = {}".format(np.array(train_iterations).shape,np.array(loss_bbox).shape)
plt.plot(train_iterations, loss_bbox, 'r-',label="loss_bbox")
plt.legend(loc='upper right')
plt.xlabel("iterations")
plt.ylabel("loss_bbox")

fig3 = plt.figure("fig3_loss_cls")
plt.plot(train_iterations, loss_cls, 'm-',label="loss_cls")
plt.legend(loc='upper right')
plt.xlabel("iterations")
plt.ylabel("loss_cls")

if rpn_cls_loss != []:
    fig4 = plt.figure("fig4_rpn_cls_loss")
    plt.plot(train_iterations, rpn_cls_loss, 'c-*',label="rpn_cls_loss")
    plt.legend(loc='upper right')
    plt.xlabel("iterations")
    plt.ylabel("rpn_cls_loss")
    fig4.show()
if rpn_loss_bbox != []:
    fig5 = plt.figure("fig5_rpn_loss_bbox")
    plt.plot(train_iterations, rpn_loss_bbox, 'k-^',label="rpn_loss_bbox")
    plt.legend(loc='upper right')
    plt.xlabel("iterations")
    plt.ylabel("rpn_loss_bbox")
    fig5.show()
plt.draw()
plt.show()
fig2.show()
fig3.show()
