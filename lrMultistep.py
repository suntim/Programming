# -*- coding: UTF-8 -*-
#!/usr/bin/env python

import matplotlib.pyplot as plt

base_lr = 0.01
gamma = 0.8
multistepsize = [50,85]
max_step = 100
lr = []
train_iter = []

multistepsize.append(max_step)
print multistepsize
for i in xrange(max_step):
    for index,stepsize in enumerate(multistepsize):
        if i <= stepsize:
            current_lr = base_lr *pow(gamma, i / stepsize)
            base_lr = current_lr
            print current_lr
            lr.append(current_lr)
            train_iter.append(i)
            print "{}--->{}".format(i,stepsize)
            break


plt.plot(train_iter,lr,'b-o')
plt.xlabel("train_iters")
plt.ylabel("lr")
plt.title("Step [gamma = %s stepsize= %s]"%(gamma,multistepsize))
plt.show()
