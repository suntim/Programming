#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
import matplotlib
matplotlib.use('pdf')
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
from fast_rcnn import myTools
from lxml import etree
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import os
from os import listdir
import re
from os.path import isfile, join

CLASSES = ('__background__', # always index 0
                        'breakage', 'crack','waterdrop','spot')

NETS = {'vgg16': ('VGG16',
                  'glass_vgg16_faster_rcnn_iter_9600.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel'),
	'myzf': ('ZF', 'ZF_faster_rcnn_final.caffemodel'),
	'resnet_18': ('ResNet-18-model', 'res_18_faster_rcnn_final.caffemodel'),
	'mobile_net': ('Mobile', 'mobile_faster_rcnn_final.caffemodel'),
	'resnet': ('ResNet-50-model', 'glass_res50_faster_rcnn_iter_11200.caffemodel')}

def object_info(data, objName, xmin_v, ymin_v, xmax_v, ymax_v):
    object = etree.SubElement(data, 'object')
    object_name = etree.SubElement(object, 'name')
    object_name.text = objName
    pose = etree.SubElement(object, 'pose')
    pose.text = 'Unspecified'
    truncated = etree.SubElement(object, 'truncated')
    truncated.text = '0'
    difficult = etree.SubElement(object, 'difficult')
    difficult.text = '0'
    bndbox = etree.SubElement(object, 'bndbox')
    xmin = etree.SubElement(bndbox, 'xmin')
    xmin.text =str(int(xmin_v))
    ymin = etree.SubElement(bndbox, 'ymin')
    ymin.text = str(int(ymin_v))
    xmax = etree.SubElement(bndbox, 'xmax')
    xmax.text = str(int(xmax_v))
    ymax = etree.SubElement(bndbox, 'ymax')
    ymax.text = str(int(ymax_v))

def vis_detections(data,im_h,im_w,im_depth,image_name, save_path, ax, class_name, dets, thresh=0.3):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return


    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        # 7 object
        object_info(data, objName=class_name, xmin_v=bbox[0], ymin_v=bbox[1], xmax_v=bbox[2], ymax_v=bbox[3])
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )

        ax.text(bbox[0], bbox[1] - 2,
                 '{:s} {:.3f}'.format(class_name, score),
                 bbox=dict(facecolor='blue', alpha=0.5),
                 fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,thresh),fontsize=14)


    dataxml = etree.tostring(data, pretty_print=True, encoding="UTF-8", method="xml", xml_declaration=True,
                             standalone=None)

    #write xml  
    with open(os.path.join(save_path,(image_name+'.xml')), 'w') as fp:
        fp.write(dataxml)
    print save_path

    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(net, image_name, save_path):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)
    im_h,im_w,im_depth = im.shape[:3]

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.2
    NMS_THRESH = 0.1

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    data = etree.Element("annotation")
    data.set('verified', 'no')
    # 1 folder
    interface_folder = etree.SubElement(data, 'folder')
    interface_folder.text = 'testXT'
    # 2 filename
    filename_txt = image_name
    filename = etree.SubElement(data, 'filename')
    filename.text = filename_txt
    # 3 path
    pathNode = etree.SubElement(data, 'path')
    pathNode.text = save_path+filename_txt+'.jpg'
    # 4 source
    source = etree.SubElement(data, 'source')
    database = etree.SubElement(source, 'database')
    database.text = 'Unknown'
    # 5 img size
    imgsize = etree.SubElement(data, 'size')
    img_width = etree.SubElement(imgsize, 'width')
    img_width.text = str(im_w)
    img_height = etree.SubElement(imgsize, 'height')
    img_height.text = str(im_h)
    img_depth = etree.SubElement(imgsize, 'depth')
    img_depth.text = str(im_depth)
    # 6 segmented
    segmented = etree.SubElement(data, 'segmented')
    segmented.text = '0'


    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(data,im_h,im_w,im_depth,image_name.split('/')[-1].split('.')[0],save_path,ax, cls, dets, thresh=CONF_THRESH)

    output_dir = os.path.join(cfg.ROOT_DIR,'output_glass',image_name.split('/')[-1] + "_detect_rst.jpg")

    plt.savefig(output_dir)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net_test', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')


    default_modeldir = os.path.join(cfg.ROOT_DIR, '..', 'models', 'pascal_voc')
    parser.add_argument('--models_dir', dest='model_dir', help='Model to use [pascal_voc]',
                        default=default_modeldir)

    default_dir = os.path.join(cfg.ROOT_DIR, '..', 'datasets', 'test')
    parser.add_argument('--dataset', dest='dataset_dir',
            		 help='dataset_dir',
            		 default=default_dir)


    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()
    print "********************cfg.MODELS_DIR[%s]*****************"%cfg.MODELS_DIR
    print "********************cfg.ROOT_DIR[%s]*****************"%cfg.ROOT_DIR
    print os.path.join(cfg.ROOT_DIR,args.model_dir)
    prototxt = os.path.join(os.path.join(cfg.ROOT_DIR,args.model_dir), NETS[args.demo_net][0],
                            'faster_rcnn_end2end', 'test.prototxt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    #mypath = os.path.join(cfg.ROOT_DIR, '..', args.dataset_dir)
    mypath = os.path.join(cfg.ROOT_DIR, args.dataset_dir)
    print "mypath = ****%s****"%mypath
    print '\n\nLoaded network {:s}'.format(caffemodel)

    if not os.path.exists(os.path.join(cfg.ROOT_DIR,'output_glass')):
         os.mkdir(os.path.join(cfg.ROOT_DIR, 'output_glass'))

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    print(mypath)


    im_names = myTools.joinPic(mypath)
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name, mypath)
