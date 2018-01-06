# -*- coding: UTF-8 -*-
#!/usr/bin/env python

import matplotlib.pyplot as plt

base_lr = 0.01
gamma = 0.1
lr = []
train_iter = []

for i in xrange(1000):
    current_lr = base_lr*pow(gamma,i)
    lr.append(current_lr)
    train_iter.append(i)

plt.plot(train_iter,lr,'b-')
plt.xlabel("train_iters")
plt.ylabel("lr")
plt.title("Exp [gamma = %s ]"%gamma)
plt.show()
