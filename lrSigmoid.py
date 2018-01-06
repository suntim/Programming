# -*- coding: UTF-8 -*-
#!/usr/bin/env python


import matplotlib.pyplot as plt
import numpy as np
import math

base_lr = 0.01
step_size = 500
gamma = 0.01
lr =[]
train_iter = []

for i in xrange(1000):

    current_lr = 1.0/(1+math.exp(-gamma*1.0*(i-step_size)))*base_lr
    lr.append(current_lr)
    train_iter.append(i)

plt.plot(train_iter,lr,'b-')
plt.xlabel("train_iters")
plt.ylabel("lr")
plt.title("Sigmoid [step_size = %s ]"%step_size)
# plt.axis([0, 10000, 0.0099, 0.01])
plt.show()
