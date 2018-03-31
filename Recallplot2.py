# -*- coding: UTF-8 -*-
#!/usr/bin/env python
import matplotlib.pyplot as plt
import pprint, pickle
path = '/home/alex/Documents/py-R-FCN/output/faster_rcnn_end2end/voc_2007_test/vgg16_faster_rcnn_iter_80000/bird_pr.pkl'
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
# plt.plot(recall,precision,'b-+')
plt.step(recall,precision,color='b',alpha=0.5,where='post')
plt.fill_between(recall,precision,step='post',alpha=0.2,color='m')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('className = {} , AP = {}'.format(str(path).split('/')[-1].split('.')[0],info['ap']))
plt.show()
