# Programming

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

  
