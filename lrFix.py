# -*- coding: UTF-8 -*-
#!/usr/bin/env python


import matplotlib.pyplot as plt

base_lr = 0.01
maxiter = 1000
lr = []
train_iter = []

for i in xrange(maxiter):
    current_lr = base_lr
    lr.append(current_lr)
    train_iter.append(i)

plt.plot(train_iter,lr,'b-')
plt.xlabel("train_iters")
plt.ylabel("lr")
plt.title("Fixed [base_lr = %s ]"%base_lr)
# plt.axis([0, 10000, 0.0099, 0.01])
plt.show()
