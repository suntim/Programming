# -*- coding: UTF-8 -*-
#!/usr/bin/env python
import numpy as np
import os
import cPickle
import xml.etree.ElementTree as ET

classses = ['__background__','breakage','crack','waterdrop','spot']
num_classes = len(classses)

def _get_glass_results_file_template():
    filename = str('comp4_det_test'+'_{:s}.txt')
    path = os.path.join('Glass2017','Main',filename)
    return path

def _load_image_set_index(image_set_file):
    with open(image_set_file)as f:
        image_index = [x.strip() for x in f.readlines()]
    return image_index

def parse_rec(filename):
    """
    一个xml所有的objects信息
    :param filename:
    :return:
    """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)
    return objects

def voc_ap(rec,prec):
    use_07_metric = False
    # 11 point metric
    if use_07_metric:
        ap =0.
        for t in np.arange(0.,1.1,0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += ap + p/11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.],rec,[1.]))#首尾数组拼接数字
        mpre = np.concatenate(([0.],prec,[0.]))

        #compute the precision  envelope
        for i in range(mpre.size-1,0,-1):
            mpre[i-1] = np.maximum(mpre[i-1],mpre[i])

        #to calculate area under PR curve, look for points
        #where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
        #sum (\Delta recall)*prec
        ap = np.sum((mrec[i+1]-mrec[i])*mpre[i+1])
    return ap



def voc_eval(filename,annopath,image_set_file,classname,cachedir,ovthresh=0.5):
    cachefile = os.path.join(cachedir,'annots.pkl')
    with open(image_set_file,'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    if not os.path.isfile(cachefile):
        print "os.path.isfile(cachefile) == False"
        recs = {}
        for i,imagename in enumerate(imagenames):
            print "annopath.format(imagename) = %s"%(annopath.format(imagename))
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i%2 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(i+1,len(imagenames))
        #save gt annots.pkl
        print 'Saving cached annotations to {}'.format(cachefile)
        with open(cachefile,'w') as f:
            cPickle.dump(recs,f)
    else:
        with open(cachefile,'r') as f:
            recs = cPickle.load(f)
        assert cachefile,'exist!!!'
    #extract gt objects for this class
    class_recs = {}
    npos = 0
    print '**recs** = ',recs #gt
    for imagename in imagenames:
        print 'imagename',imagename
        R = [obj for obj in recs[imagename] if obj['name'] == classname]#所有的一个类别，提取它的坐标
        print "R = {}".format(R)
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False]*len(R)
        npos = npos + sum(~difficult)#number of positive
        print "len(R) = {}, npos = {}".format(len(R),npos)#number of positive
        class_recs[imagename] = {'bbox':bbox,'difficult':difficult,'det':det}

    # print "class_recs = {}".format(class_recs)
    #read dets
    detfile = filename.format(classname)
    with open(detfile,'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids  = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    #sort by confidence
    sorted_ind = np.argsort(-confidence)#数组值从小到大的索引值,加“-”从大到小
    # print "sorted_ind = {}".format(sorted_ind)
    sorted_scores = np.sort(-confidence)

    if BB != []:
        BB = BB[sorted_ind,:]
    image_ids = [image_ids[x] for x in sorted_ind]

    #go down dets and mark TPs and FPs
    nd = len(image_ids)
    print "nd = {}".format(nd)
    tp = np.zeros(nd)#True Positive
    fp = np.zeros(nd)#False Positive误检的
    for d in range(nd):
        print "当前检测图片：【{}】".format(image_ids[d])
        R = class_recs[image_ids[d]]
        bb = BB[d,:].astype(float)
        print "bb = {}".format(bb)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            #compute overlaps
            #intersection
            print "BBGT[:] = {},GT number = {}".format(BBGT[:],len(BBGT[:,0]))
            ixmin = np.maximum(BBGT[:,0],bb[0])
            iymin = np.maximum(BBGT[:,1],bb[1])
            ixmax = np.minimum(BBGT[:,2],bb[2])
            iymax = np.minimum(BBGT[:,3],bb[3])
            iw = np.maximum(ixmax-ixmin+1.,0.)
            ih = np.maximum(iymax-iymin+1.,0.)
            inters = iw*ih

            #union
            uni = ((bb[2]-bb[0]+1.)*(bb[3]-bb[1]+1.)+
                   (BBGT[:,2]-BBGT[:,0]+1.)*(BBGT[:,3]-BBGT[:,1]+1.)-
                   inters)
            overlaps = inters/uni
            print ">>【overlaps】: {}".format(overlaps)
            ovmax = np.max(overlaps)
            print ">>ovmax:{}".format(ovmax)
            jmax = np.argmax(overlaps)#索引值
            print ">>jmax:{}".format(jmax)
        if ovmax > ovthresh:
            print "ovmax = %s"%ovmax
            # print "R['difficult'][3]={}".format(R['difficult'][3])
            if not R['difficult'][jmax]:#非难检
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    print "重复检测！"
                    fp[d] = 1.#重复检测
            print "[R] = {}".format(R)
        else:
            fp[d] = 1.
    #compute precision recall
    print "{} 中：False Positive = {}".format(classname,fp)
    fp = np.cumsum(fp)
    print "{} 中：累计False Positive = {}".format(classname,fp)
    print "{} 中：True Positive = {}".format(classname, tp)
    tp = np.cumsum(tp)
    print "{} 中：累计True Positive = {}".format(classname, tp)
    rec = tp/float(npos)
    #avoid divide by zero in case the first detection matches a difficult

    #ground truth
    prec = tp/np.maximum(tp+fp,np.finfo(np.float64).eps)#0< np.finfo(np.float64).eps=2.22044604925e-16 <0.1
    ap = voc_ap(rec,prec)

    if nd == 0:
        fn = npos - 0  # 漏检的
    else:
        fn = npos - tp # 漏检的
    return rec,prec,fn,fp,ap



if __name__ == '__main__':
    annopath = os.path.join(r'C:\Users\xuting7\Desktop\gt', '{}.xml')#GT Xml路径
    print 'annopath = %s'%annopath
    image_set_file = r'C:\Users\xuting7\Desktop\test.txt'#图片名字，不包含后缀名
    cachedir = os.path.join('Glass2017', 'annotations_cache')
    if not os.path.exists(cachedir):
        os.mkdir(cachedir)
    aps = []
    assert os.path.exists(image_set_file),'path does not exist!'#false,才执行
    img_index = _load_image_set_index(image_set_file)

    for i,cls in enumerate(classses):
        if cls == '__background__':
            continue
        filename = _get_glass_results_file_template().format(cls)
        print "======================================cls = %s=============================================="%cls
        rec,prec,fn,fp,ap = voc_eval(filename,annopath,image_set_file,cls,cachedir,ovthresh=0.1)
        aps += [ap]
        print "~~~~~{} : recall = {} prec = {} 漏检fn = {} 误检fp = {} AP = {}".format(cls,rec,prec,fn,fp,ap)
        with open((cls+'_precisionRecall.pkl'),'w') as f:
            cPickle.dump({'rec':rec,'prec':prec,'ap':ap},f)
    print ('Aps = {}'.format(aps))
    print ('Mean AP = {:.4f}'.format(np.mean(aps)))


