# -*- coding: UTF-8 -*-
#!/usr/bin/env python
from pylab import matplotlib,mpl
# mpl.rcParams['font.sans-serif'] = ['SimHei']            #SimHei是黑体的意思
# mpl.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import re
import numpy as np

zhfont = matplotlib.font_manager.FontProperties(fname = "/usr/share/fonts/truetype/arphic/ukai.ttc")
# read the log file
# fp = open('/home/alex/Documents/py-R-FCN/experiments/logs/faster_rcnn_end2end_VGG16_.txt.2018-03-05_21-45-13', 'r')
fp = open('/home/alex/Pictures/FASTER-RCNN-OHEM/OHEM_MUTI-1-1-1-1/2faster_rcnn_end2end_VGG16_.txt.2018-03-05_21-45-13.txt', 'r')
train_iterations = []
train_loss = []
loss_bbox = []
loss_cls = []
rpn_cls_loss = []
rpn_loss_bbox = []

for ln in fp:
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

def plotSmooth(train_iterations, train_loss,kernel_size = 100,stride = 10):
    train_loss_mean = []
    train_loss_mean.append(train_loss[0])
    train_iterations_mean = []
    train_iterations_mean.append(0)
    for tup in (train_iterations[z:z+kernel_size] for z in range(0,len(train_iterations),stride)):
        train_iterations_mean.append(int(np.mean(tup)))
    for tup in (train_loss[z:z+kernel_size] for z in range(0,len(train_loss),stride)):
        train_loss_mean.append(np.mean(tup))
    return train_iterations_mean,train_loss_mean

fig1 = plt.figure("fig1_totall_loss")
# plot curves
plt.plot(train_iterations, train_loss, 'b-',label="train loss")
train_iterations_mean,train_loss_mean = plotSmooth(train_iterations, train_loss,kernel_size = 100,stride = 10)
plt.plot(train_iterations_mean,train_loss_mean,'y-o',markersize=2,label=u'移动平均')
plt.title(u'训练的总损失',fontproperties=zhfont) #指定字体
plt.xlabel("iterations")
plt.ylabel("loss")
# set location of the legend,
# 1->rightup corner, 2->leftup corner, 3->leftdown corner
# 4->rightdown corner, 5->rightmid ...
plt.legend(loc=1,prop =zhfont)


fig2 = plt.figure("fig2_loss_bbox")
print "train_iterations.shape = {} , loss_bbox.shape = {}".format(np.array(train_iterations).shape,np.array(loss_bbox).shape)
plt.plot(train_iterations, loss_bbox, 'r-',label="loss_bbox")
train_iterations_mean,train_loss_mean = plotSmooth(train_iterations, loss_bbox,kernel_size = 100,stride = 10)
plt.plot(train_iterations_mean,train_loss_mean,'y-o',markersize=2,label=u'移动平均')
plt.title(u'边框回归的损失',fontproperties=zhfont) #指定字体
plt.legend(loc='upper right',prop=zhfont)
plt.xlabel("iterations")
plt.ylabel("loss_bbox")

fig3 = plt.figure("fig3_loss_cls")
plt.plot(train_iterations, loss_cls, 'm-',label="loss_cls")
train_iterations_mean,train_loss_mean = plotSmooth(train_iterations, loss_cls,kernel_size = 100,stride = 10)
plt.plot(train_iterations_mean,train_loss_mean,'y-o',markersize=2,label=u'移动平均')
plt.title(u'分类的损失',fontproperties=zhfont) #指定字体
plt.legend(loc='upper right',prop=zhfont)
plt.xlabel("iterations")
plt.ylabel("loss_cls")

if rpn_cls_loss != []:
    fig4 = plt.figure("fig4_rpn_cls_loss")
    plt.plot(train_iterations, rpn_cls_loss, 'c-*',label="rpn_cls_loss")
    train_iterations_mean, train_loss_mean = plotSmooth(train_iterations, rpn_cls_loss, kernel_size=100, stride=10)
    plt.plot(train_iterations_mean, train_loss_mean, 'y-o', markersize=2, label=u'移动平均')
    plt.title(u'RPN层分类的损失', fontproperties=zhfont)  # 指定字体
    plt.legend(loc='upper right',prop=zhfont)
    plt.xlabel("iterations")
    plt.ylabel("rpn_cls_loss")
    fig4.show()
if rpn_loss_bbox != []:
    fig5 = plt.figure("fig5_rpn_loss_bbox")
    plt.plot(train_iterations, rpn_loss_bbox, 'k-^',label="rpn_loss_bbox")
    train_iterations_mean, train_loss_mean = plotSmooth(train_iterations, rpn_loss_bbox, kernel_size=100, stride=10)
    plt.plot(train_iterations_mean, train_loss_mean, 'y-o', markersize=2, label=u'移动平均')
    plt.title(u'RPN层边框回归的损失', fontproperties=zhfont)  # 指定字体
    plt.legend(loc='upper right',prop=zhfont)
    plt.xlabel("iterations")
    plt.ylabel("rpn_loss_bbox")
    fig5.show()

plt.draw()
plt.show()
