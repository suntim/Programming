# -*- coding: UTF-8 -*-
#!/usr/bin/env python
import matplotlib.pyplot as plt

base_lr = 0.01
gamma = 0.8
stepsize = 1000
lr = []
train_iter = []

for i in xrange(2000):
    current_lr = base_lr*pow(gamma,i/stepsize)
    lr.append(current_lr)
    train_iter.append(i)

plt.plot(train_iter,lr,'b-o')
plt.xlabel("train_iters")
plt.ylabel("lr")
plt.title("Step [gamma = %s stepsize= %s]"%(gamma,stepsize))
plt.show()
