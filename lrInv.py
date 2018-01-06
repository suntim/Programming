# -*- coding: UTF-8 -*-
#!/usr/bin/env python

import matplotlib.pyplot as plt

base_lr = 0.01
gamma = 0.05
power = 0.5
lr = []
train_iter = []

for i in xrange(20000):
    current_lr = base_lr*pow(1+gamma*i,-power)
    lr.append(current_lr)
    train_iter.append(i)

plt.plot(train_iter,lr,'b-')
plt.xlabel("train_iters")
plt.ylabel("lr")
plt.title("Inv [gamma = %s power= %s]"%(gamma,power))
plt.show()
