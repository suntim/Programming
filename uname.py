# -*- coding: UTF-8 -*-
#!/usr/bin/env python
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']            #SimHei是黑体的意思
import matplotlib.pyplot as plt
from pylab import *

Model_Name = [u'混沌期',u'启蒙期',u'起步期',u'成长期',u'成熟期']
Average_Forward_pass = [64.8538, 72.4793, 232.038, 792.83, 1167.5]
Average_Backward_pass = [73.269, 81.0171, 220.578, 751.987, 4068.0]
Average_Forward_Backward = [139.431, 155.33, 455.283, 1544.817, 5235.5]
Total_Time = [6971.57, 7766.52, 22764.2, 77437.4, 261775]

plt.figure('fg1')
plt.plot(Average_Forward_pass,'b-*')
plt.plot(Average_Backward_pass,'r-^')
plt.plot(Average_Forward_Backward,'c-h')
x=np.linspace(0,len(Average_Forward_pass)-1,len(Average_Forward_pass))
plt.xlabel('Models' ,fontproperties='SimHei')
plt.xticks(x,Model_Name)
plt.ylabel('ms/Time')
plt.legend(['Forward','Backward','Forward_Backward'])
plt.show('fg1')

plt.figure('fg2')
plt.plot(Total_Time,'m-+')
x=np.linspace(0,len(Average_Forward_pass)-1,len(Average_Forward_pass))
# plt.xlabel(Model_Name ,fontproperties='SimHei')
plt.xticks(x,Model_Name)
plt.xlabel(u'阶段' ,fontproperties='SimHei')
plt.ylabel(u'能力',fontproperties='SimHei')
plt.legend([u'学习能力曲线'])
plt.show('fg2')
