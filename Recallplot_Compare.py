# -*- coding: UTF-8 -*-
#!/usr/bin/env python
import matplotlib.pyplot as plt
import pprint, pickle
path1 = '/home/alex/Pictures/Pickl/vgg16_faster_rcnn_5iter_60000/person_pr.pkl'
path2 = '/home/alex/Pictures/Pickl/best_70000/person_pr.pkl'

pkl_file = open(path1, 'rb')
info1 = pickle.load(pkl_file)
# pprint.pprint(info1)
precision1 = info1['prec']
recall1= info1['rec']
print len(precision1)
print len(recall1)
pkl_file.close()

pkl_file = open(path2, 'rb')
info2 = pickle.load(pkl_file)
# pprint.pprint(info2)
precision2 = info2['prec']
recall2= info2['rec']
print len(precision2)
print len(recall2)
pkl_file.close()

plt.figure('fig1')
plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
# plt.plot(recall,precision,'b-+')
plt.step(recall1,precision1,alpha=0.5,linestyle=':',where='post',label='Orig_NMS AP = {:.3f}'.format(info1['ap']))
plt.fill_between(recall1,precision1,step='post',alpha=0.2,color='b')

plt.step(recall2,precision2,alpha=0.5,where='post',label='Improve_NMS AP = {:.3f}'.format(info2['ap']))
plt.fill_between(recall2,precision2,step='post',alpha=0.2,color='m')

plt.legend(loc=1)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('className = {}'.format(str(path1).split('/')[-1].split('.')[0]))
plt.show()
