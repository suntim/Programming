# -*- coding: UTF-8 -*-
#!/usr/bin/env python
import matplotlib.pyplot as plt
import pprint, pickle
path = 'C:/Users/xuting7/Downloads/breakage_pr.pkl'
pkl_file = open(path, 'rb')

info = pickle.load(pkl_file)
pprint.pprint(info)
precision = info['prec']
recall= info['rec']
print len(precision)
print len(recall)
pkl_file.close()

plt.figure('fig1')
plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
plt.plot(recall,precision,'b-+')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('className = {} , AP = {}'.format(str(path).split('/')[-1].split('.')[0],info['ap']))
plt.show()
