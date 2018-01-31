# -*- coding: UTF-8 -*-
#!/usr/bin/env python
import matplotlib.pyplot as plt
from pylab import *

Model_Name = ['VGG16','Res18','Res50','Res101']
Average_Forward_pass = [64.8538,72.4793,232.038,792.83]
Average_Backward_pass = [73.269,81.0171,220.578,751.987]
Average_Forward_Backward = [139.431,155.33,455.283,1544.817]
Total_Time = [6971.57,7766.52,22764.2,77437.4]

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
plt.xlabel('Models' ,fontproperties='SimHei')
plt.ylabel('ms/Time')
plt.legend(['Total_Time'])
plt.show('fg2')
