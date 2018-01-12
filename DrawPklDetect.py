# -*- coding: UTF-8 -*-
#!/usr/bin/env python

import os.path
import matplotlib.pyplot as plt
import cv2
import cPickle as pickle

f = open(r'C:\Users\x\Downloads\annots.pkl')
info = pickle.load(f)

file_path_img = r'C:\Users\x\Downloads\MyPython\PlotLrFigure\PlotResult\oldTrainTest'
save_file_path = './pkl'



for line in info:
    # print line
    # print info[line]
    # print len(info[line])
    if len(info[line]) == 0:
        continue
    # img = Image.open(os.path.join(file_path_img, line + '.jpg'))
    im = cv2.imread(os.path.join(file_path_img, line + '.jpg'))
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for index in xrange(len(info[line])):
        class_name = info[line][index]['name']
        print class_name
        bbox = info[line][index]['bbox']
        print bbox
        ax.add_patch(plt.Rectangle((bbox[0], bbox[1]),bbox[2] - bbox[0],bbox[3] - bbox[1], fill=False,edgecolor='blue', linewidth=2))
        ax.text(bbox[0], bbox[1] - 2,'{:s}'.format(class_name),bbox=dict(facecolor='m', alpha=0.5),fontsize=14, color='white')
        # ax.set_title(('{} detections with p({} | box) >= {:.1f}').format(class_name, class_name, 0.3),fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.draw()
        output_dir = os.path.join(save_file_path, line+ "_detect_rst.jpg")
    plt.savefig(output_dir)
