# Programming
1、demoConveOut.py
'''
#!/usr/bin/env python
#encoding=utf8
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
matplotlib.use('Agg')
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
from os import listdir
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import re
CLASSES = ('__background__',
		   'aeroplane', 'bicycle', 'bird', 'boat',
		   'bottle', 'bus', 'car', 'cat', 'chair',
		   'cow', 'diningtable', 'dog', 'horse',
		   'motorbike', 'person', 'pottedplant',
		   'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
				  'vgg16_faster_rcnn_iter_70000.caffemodel'),
		'zf': ('ZF',
				  'ZF_faster_rcnn_final.caffemodel')}


#增加ax参数
def vis_detections(class_name, dets, ax, thresh=0.5):
	"""Draw detected bounding boxes."""
	inds = np.where(dets[:, -1] >= thresh)[0]
	if len(inds) == 0:
		return
# 删除这三行
#     im = im[:, :, (2, 1, 0)]
#     fig, ax = plt.subplots(figsize=(12, 12))
#     ax.imshow(im, aspect='equal')
	for i in inds:
		bbox = dets[i, :4]
		score = dets[i, -1]

		ax.add_patch(
			plt.Rectangle((bbox[0], bbox[1]),
						  bbox[2] - bbox[0],
						  bbox[3] - bbox[1], fill=False,
						  edgecolor='red', linewidth=1) # 矩形线宽从3.5改为1
			)
		ax.text(bbox[0], bbox[1] - 2,
				'{:s} {:.3f}'.format(class_name, score),
				bbox=dict(facecolor='blue', alpha=0.5),
				fontsize=14, color='white')

	ax.set_title(('{} detections with '
				  'p({} | box) >= {:.1f}').format(class_name, class_name,
												  thresh),
				  fontsize=14)
def save_feature_pic(data,name,image_name=None, padsize=1,padval=1):
       data=data[0]
       n=int(np.ceil(np.sqrt(data.shape[0])))
       padding=((0,n**2-data.shape[0]),(0,0),(0,padsize))+((0,0),)*(data.ndim-3)
       data=np.pad(data,padding,mode='constant',constant_values=(padval,padval))

       data=data.reshape((n,n)+data.shape[1:]).transpose((0,2,1,3)+tuple(range(4,data.ndim+1)))
       data=data.reshape((n*data.shape[1],n*data.shape[3])+data.shape[4:])
       plt.figure()
       plt.imshow(data,cmap='gray')
       plt.axis('off')

       if image_name==None:
           print"image_name is None"
       else:
           out_feature_dir=os.path.join(cfg.ROOT_DIR,'My_feature_picture',image_name.split('/')[-1].split('.')[0])
           print"out_feature_dir=****{}****".format(out_feature_dir)
           check_file(out_feature_dir)
           plt.savefig(os.path.join(out_feature_dir,name+".jpg"),dpi=400,bbox_inches="tight")

def check_file(path):
      if not os.path.exists(path):
          os.makedirs(path)



def demo(net, image_name):
	"""Detect object classes in an image using pre-computed object proposals."""

	# Load the demo image
	im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
	im = cv2.imread(im_file)

	# Detect all object classes and regress object bounds
	timer = Timer()
	timer.tic()
	scores, boxes = im_detect(net, im)
	timer.toc()
	print ('Detection took {:.3f}s for '
		   '{:d} object proposals').format(timer.total_time, boxes.shape[0])

        for k,v in net.blobs.items():
            if k.find("conv")>-1 or k.find("pool")>-1 or k.find("rpn")>-1:
                save_feature_pic(v.data,k.replace("/",""),image_name)#(net.blobs["conv1_1"].data,"conv1_1")

	# Visualize detections for each class
	CONF_THRESH = 0.7
	NMS_THRESH = 0.3
	# 将vis_detections 函数中for 循环之前的3行代码移动到这里
	im = im[:, :, (2, 1, 0)]
	fig,ax = plt.subplots(figsize=(12, 12))
	ax.imshow(im, aspect='equal')
	for cls_ind, cls in enumerate(CLASSES[1:]):
		cls_ind += 1 # because we skipped background
		cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
		cls_scores = scores[:, cls_ind]
		dets = np.hstack((cls_boxes,
						  cls_scores[:, np.newaxis])).astype(np.float32)
		keep = nms(dets, NMS_THRESH)
		dets = dets[keep, :]
		#将ax做为参数传入vis_detections
		vis_detections(cls, dets, ax,thresh=CONF_THRESH)
	# 将vis_detections 函数中for 循环之后的3行代码移动到这里
	plt.axis('off')
	plt.tight_layout()
	plt.draw()
	output_dir = os.path.join(cfg.ROOT_DIR,'output_my',
	image_name.split('/')[-1] + "_detect_rst.jpg")

	plt.savefig(output_dir)

def parse_args():
	"""Parse input arguments."""
	parser = argparse.ArgumentParser(description='Faster R-CNN demo')
	parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
						default=0, type=int)
	parser.add_argument('--cpu', dest='cpu_mode',
						help='Use CPU mode (overrides --gpu)',
						action='store_true')
	parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
						choices=NETS.keys(), default='vgg16')
	default_dir = os.path.join(cfg.ROOT_DIR,'..','datasets','test')
	
	parser.add_argument('--dataset',dest='dataset_dir',help='dataset_dir',
						default=default_dir)
	args = parser.parse_args()
	return args
def print_param(net):
        print"----------net.blobs.items-------------"
        for k,v in net.blobs.items():
            print (k,v.data.shape)
        print"-----------net.param-----------"
        for k,v in net.params.items():
            print(k,v[0].data.shape)

if __name__ == '__main__':
	cfg.TEST.HAS_RPN = True  # Use RPN for proposals

	args = parse_args()

        #这里模型训练类型是写死的吗？
#	prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
#							'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
        #除experiment的scripts的myfaster_rcnn_end2end改test.prototxt还有这里？
	prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
							'faster_rcnn_end2end', 'test.prototxt')
        #这里放模型的参数结果路径
	caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
							  NETS[args.demo_net][1])

	if not os.path.isfile(caffemodel):
		raise IOError(('{:s} not found.\nDid you run ./data/scripts/'
					   'fetch_faster_rcnn_models.sh?').format(caffemodel))

	if args.cpu_mode:
		caffe.set_mode_cpu()
	else:
		caffe.set_mode_gpu()
		caffe.set_device(args.gpu_id)
		cfg.GPU_ID = args.gpu_id
	net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        print '\n\nnet={:s}'.format(net)
        print_param(net)
	print '\n\nLoaded network {:s}'.format(caffemodel)
        print "cfg.MODELS_DIR=****{}****".format(cfg.MODELS_DIR)
	# Warmup on a dummy image
	im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
	for i in xrange(2):
		_, _= im_detect(net, im)

	my_data_path = os.path.join(cfg.ROOT_DIR, args.dataset_dir)
	print 'my_data_path:{:s}'.format(my_data_path)
	if not os.path.exists(os.path.join(cfg.ROOT_DIR,'output_my')):
		os.mkdir(os.path.join(cfg.ROOT_DIR, 'output_my'))
	im_names = [os.path.join(my_data_path, f) for f in listdir(my_data_path) if re.match(r'.*\.jpg', f)]
	for im_name in im_names:
		print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
		print 'Demo for data/demo/{}'.format(im_name)
		demo(net, im_name)

	im_names = [os.path.join(my_data_path, f) for f in listdir(my_data_path) if re.match(r'.*\.jpeg', f)]
	for im_name in im_names:
		print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
		print 'Demo for data/demo/{}'.format(im_name)
		demo(net, im_name)

	im_names = [os.path.join(my_data_path, f) for f in listdir(my_data_path) if re.match(r'.*\.bmp', f)]
	for im_name in im_names:
		print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
		print 'Demo for data/demo/{}'.format(im_name)
		demo(net, im_name)
	#plt.show()
'''

2、bamboo_test_my.py
'''
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
                        'external', 'inner')

NETS = {'vgg16': ('VGG16',
                  'bamboo_ohem_vgg16_frcnn_iter_6000.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel'),
	'myzf': ('ZF', 'ZF_faster_rcnn_final.caffemodel'),
	'resnet_18': ('ResNet-18-model', 'res_18_faster_rcnn_final.caffemodel'),
	'mobile_net': ('Mobile', 'mobile_faster_rcnn_final.caffemodel'),
	'resnet': ('ResNet-50-model', 'res_faster_rcnn_iter_10000.caffemodel')}


def vis_detections(ax, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    # im = im[:, :, (2, 1, 0)]
    # fig, ax = plt.subplots(figsize=(12, 12))
    # ax.imshow(im, aspect='equal')
    # print('inds')
    # print(inds)
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s}'.format(class_name),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
        # ax.text(bbox[0], bbox[1] - 2,
                # '{:s} {:.3f}'.format(class_name, score),
                # bbox=dict(facecolor='blue', alpha=0.5),
                # fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.3
    NMS_THRESH = 0.1

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    #cv2.waitKey()

    # print('scores')
    # print(scores.size)
    # print(scores)
    # print('boxes')
    # print(boxes.size)
    # print(boxes)

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(ax, cls, dets, thresh=CONF_THRESH)

    # print('keep')
    # print(keep)
    output_dir = os.path.join(cfg.ROOT_DIR,'output_bamboo',
      image_name.split('/')[-1] + "_detect_rst.jpg")

    plt.savefig(output_dir)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
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
                            'faster_rcnn_end2end', 'test_bamboo.prototxt')
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

    if not os.path.exists(os.path.join(cfg.ROOT_DIR,'output_bamboo')):
         os.mkdir(os.path.join(cfg.ROOT_DIR, 'output_bamboo'))

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    print(mypath)


    im_names = myTools.joinPic(mypath)
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name)
'''


3、bamboo_mAP.sh
'''
./experiments/scripts/bamboo_mAP.sh 0 VGG16 bamboo
'''

4、bamboo_mAP2.sh
'''
./tools/test_net.py\
        --gpu 0\
        --def models/bamboo/VGG16/faster_rcnn_end2end/test_bamboo.prototxt\
        --net data/faster_rcnn_models/bamboo_ohem_vgg16_frcnn_iter_6000.caffemodel\
        --imdb bamboo_2017_test\
        --cfg experiments/cfgs/faster_rcnn_end2end_finetune.yml\
'''

5、bamboo_train.sh
'''
./experiments/scripts/bamboo_faster_rcnn_end2end.sh 0 VGG16 bamboo
'''

6、bamboo_test.sh
'''
./tools/bamboo_test_my.py\
       --gpu 0\
       --net vgg16\
       --models_dir models/bamboo\
       --dataset data/bamboo_test_pic\
'''

7、bamboo_test_my.py
'''
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
                        'external', 'inner')

NETS = {'vgg16': ('VGG16',
                  'bamboo_ohem_vgg16_frcnn_iter_6000.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel'),
	'myzf': ('ZF', 'ZF_faster_rcnn_final.caffemodel'),
	'resnet_18': ('ResNet-18-model', 'res_18_faster_rcnn_final.caffemodel'),
	'mobile_net': ('Mobile', 'mobile_faster_rcnn_final.caffemodel'),
	'resnet': ('ResNet-50-model', 'res_faster_rcnn_iter_10000.caffemodel')}


def vis_detections(ax, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    # im = im[:, :, (2, 1, 0)]
    # fig, ax = plt.subplots(figsize=(12, 12))
    # ax.imshow(im, aspect='equal')
    # print('inds')
    # print(inds)
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s}'.format(class_name),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
        # ax.text(bbox[0], bbox[1] - 2,
                # '{:s} {:.3f}'.format(class_name, score),
                # bbox=dict(facecolor='blue', alpha=0.5),
                # fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.3
    NMS_THRESH = 0.1

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    #cv2.waitKey()

    # print('scores')
    # print(scores.size)
    # print(scores)
    # print('boxes')
    # print(boxes.size)
    # print(boxes)

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(ax, cls, dets, thresh=CONF_THRESH)

    # print('keep')
    # print(keep)
    output_dir = os.path.join(cfg.ROOT_DIR,'output_bamboo',
      image_name.split('/')[-1] + "_detect_rst.jpg")

    plt.savefig(output_dir)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
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
                            'faster_rcnn_end2end', 'test_bamboo.prototxt')
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

    if not os.path.exists(os.path.join(cfg.ROOT_DIR,'output_bamboo')):
         os.mkdir(os.path.join(cfg.ROOT_DIR, 'output_bamboo'))

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    print(mypath)


    im_names = myTools.joinPic(mypath)
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name)
'''

8、bamboo_faster_rcnn_end2end.sh
'''
#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    PT_DIR="pascal_voc"
    ITERS=20000
    ;;
  bamboo)
    TRAIN_IMDB="bamboo_2017_trainval"
    TEST_IMDB="bamboo_2017_test"
    PT_DIR="bamboo"
    ITERS=6000
    ;;
  coco)
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="coco_2014_train"
    TEST_IMDB="coco_2014_minival"
    PT_DIR="coco"
    ITERS=490000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/faster_rcnn_end2end_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${PT_DIR}/${NET}/faster_rcnn_end2end/solver_bamboo.prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel\
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/faster_rcnn_end2end_finetune.yml \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x
'''

9、lib/datasets/bamboo.py 
'''
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from voc_eval import voc_eval
from fast_rcnn.config import cfg

class bamboo(imdb):
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'bamboo_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path,'Bamboo'+self._year)
        print "*****self._devkit_path[%s]******"%self._devkit_path 
        print "**********self._data_path[%s] *********"%self._data_path
        self._classes = ('__background__', # always index 0
                         'external','inner')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._devkit_path), \
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'BambooDevkit' + self._year)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_bamboo_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if int(self._year) == 2017 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def rpn_roidb(self):
        if int(self._year) == 2017 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_bamboo_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_ind[obj.find('name').text.strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    def _get_bamboo_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            'Bamboo' + self._year,
            'Main',
            filename)
        return path

    def _write_bamboo_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = self._get_bamboo_results_file_template().format(cls)
            print "*******filename=[[[[%s]]]]"%filename 
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir = 'output'):
        annopath = os.path.join(
            self._devkit_path,
            'Bamboo' + self._year,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'Bamboo' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2050 else False
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_bamboo_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _do_matlab_eval(self, output_dir='output'):
        print '-----------------------------------------------------'
        print 'Computing results with the official MATLAB eval code.'
        print '-----------------------------------------------------'
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
               .format(self._devkit_path, self._get_comp_id(),
                       self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_bamboo_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_bamboo_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    from datasets.bamboo import bamboo 
    d = bamboo('trainval', '2017')
    res = d.roidb
    from IPython import embed; embed()
'''

10、lib/datasets/factory.py 建立data/BambooDevkit2017/Bamboo2017/Annotations/
'''
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
from datasets.railway import railway
from datasets.bamboo import bamboo
# from datasets.insulator import insulator
import numpy as np

# set up bamboo datasets
for year in ['2017']:
    for split in ('trainval', 'test'):
        name = 'bamboo_{}_{}'.format(year, split)
        __sets[name] = (lambda  split=split, year=year:bamboo(split, year))

# set up railway datasets
for year in ['20170611']:
    for split in ('trainval', 'test'):
        name = 'railway_{}_{}'.format(year, split)
        __sets[name] = (lambda  split=split, year=year:railway(split, year))

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()

'''
11、解析xml
'''
# -*- coding: UTF-8 -*-
#!/usr/bin/env python
import xml.etree.ElementTree as ET
import os
import re
COUNT = {"t-insulator":0,"b-insulator":0,"c-support":0,"ft-support":0,
         "ff-tube":0,"ff-device":0,"connector":0,"bf-tube":0,
         "bf-device":0,"i-line":0,"w-line":0}

W_name = ["t-insulator","b-insulator","c-support","ft-support",
          "ff-tube","ff-device","connector","bf-tube",
          "bf-device","i-line","w-line"]

with open('countXml.txt','w') as f:
    xml_dir = 'C:/Users/xuting7/Downloads/railway测试图片'
    for fileName in os.listdir(xml_dir):
        if re.match(".*.xml",fileName):
            # print("fileName = %s"%fileName)
            xml_path = os.path.join(xml_dir,fileName)
            #print(xml_path)
            tree = ET.parse(xml_path)
            root_name = tree.getroot()
            #遍历xml文档
            for childs in root_name:
                # print "%s ---> %s"%(childs.tag,childs.attrib)
                for chil in childs.iter('object'):
                    for name in chil.iter('name'):
                        for k in COUNT.keys():
                            #print("k=%s,name=%s"%(k,name.text))
                            if k == name.text:
                                COUNT[k] += 1

    for W_item in W_name:
        f.writelines("%s = %s \n"%(W_item,COUNT[W_item]))

print("Done!")
'''
12、解析xml2
'''
# -*- coding: UTF-8 -*-
#!/usr/bin/env python
import xml.etree.ElementTree as ET
import os
import re
import numpy as np
COUNT = np.zeros([1,11],dtype=int)

W_name = ["t-insulator","b-insulator","c-support","ft-support",
          "ff-tube","ff-device","connector","bf-tube",
          "bf-device","i-line","w-line"]

with open('countXml.txt','w') as f:
    xml_dir = 'C:/Users/xuting7/Downloads/railway测试图片'
    for fileName in os.listdir(xml_dir):
        if re.match(".*.xml",fileName):
            # print("fileName = %s"%fileName)
            xml_path = os.path.join(xml_dir,fileName)
            #print(xml_path)
            tree = ET.parse(xml_path)
            root_name = tree.getroot()
            #遍历xml文档
            for childs in root_name:
                # print "%s ---> %s"%(childs.tag,childs.attrib)
                for chil in childs.iter('object'):
                    for name in chil.iter('name'):
                        for index,W_item in enumerate(W_name):
                            #print("k=%s,name=%s"%(k,name.text))
                            if W_item == name.text:
                                COUNT[0,index] += 1

    for index,W_item in enumerate(W_name):
        f.writelines("%s = %s \n"%(W_item,COUNT[0,index]))

print("Done!")
'''
13、demo_rfcn_my_detect.sh
./tools/demo_rfcn_my.py\
       --gpu 0\
       --net ResNet-50\
       --models_dir models/pascal_voc\
       --dataset data/demo\
       
14、demo_rfcn_my.py
#!/usr/bin/env python

# --------------------------------------------------------
# R-FCN
# Copyright (c) 2016 Yuwen Xiong
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
import matplotlib
matplotlib.use('Agg')
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from fast_rcnn import myTools
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import re

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'ResNet-101': ('ResNet-101',
                  'resnet101_rfcn_final.caffemodel'),
        'ResNet-50': ('ResNet-50',
                  'resnet50_rfcn_final.caffemodel')}


def vis_detections(class_name, dets, ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    # im = im[:, :, (2, 1, 0)]
    # fig, ax = plt.subplots(figsize=(12, 12))
    # ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

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
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.draw()

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4:8]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections( cls, dets,ax, thresh=CONF_THRESH)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    output_dir = os.path.join(cfg.ROOT_DIR, 'output_my',
    image_name.split('/')[-1] + "_detect_rst.jpg")

    plt.savefig(output_dir)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [ResNet-101]',
                        choices=NETS.keys(), default='ResNet-101')

    default_modeldir = os.path.join(cfg.ROOT_DIR, '..', 'models', 'pascal_voc')
    parser.add_argument('--models_dir',dest='model_dir',help='Model_Dir to use [pascal_voc]',
                        default=default_modeldir)

    default_dir = os.path.join(cfg.ROOT_DIR, '..', 'datasets', 'test')
    parser.add_argument('--dataset', dest='dataset_dir', help='dataset_dir',
                        default=default_dir)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    print "**********cfg.MODELS_DIR = %s **********"%cfg.MODELS_DIR
    args = parse_args()

    prototxt = os.path.join(os.path.join(cfg.ROOT_DIR,args.model_dir), NETS[args.demo_net][0],
                            'rfcn_end2end', 'test_agnostic.prototxt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'rfcn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\n').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)
    print '\n\nnet={:s}'.format(net)
    print "cfg.MODELS_DIR=****{}".format(cfg.MODELS_DIR)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    mypath = os.path.join(cfg.ROOT_DIR, args.dataset_dir)
    print 'mypath:{:s}'.format(mypath)
    if not os.path.exists(os.path.join(cfg.ROOT_DIR, 'output_my')):
        os.mkdir(os.path.join(cfg.ROOT_DIR, 'output_my'))
    im_names = myTools.joinPic(mypath)
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name)


