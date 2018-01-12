# -*- coding: UTF-8 -*-
#!/usr/bin/env python
""""
使用的xml格式：
 <bndbox>
      <xmin>1586</xmin>
      <ymin>734</ymin>
      <xmax>2856</xmax>
      <ymax>1175</ymax>
 </bndbox>
"""
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as ET
import re
import cv2

xml_dir = unicode(r'C:\Users\xt\Desktop\TEST','utf-8')
img_dir = unicode(r'C:\Users\xt\Desktop\TEST','utf-8')
save_dir = r'D:\TMP\result'


for fileName in os.listdir(xml_dir):
    if re.match(".*.xml$",fileName):
        # print("fileName = %s"%fileName)
        xml_path = os.path.join(xml_dir,fileName)
        tree = ET.parse(xml_path)
        root_name = tree.getroot()
        # 遍历xml文档
        for childs in root_name:
            for fName in childs.iter('filename'):
                imgName = fName.text+".jpg"
                img = cv2.imread(os.path.join(img_dir,imgName))
                h,w = img.shape[:2]
                # print h,w
                # reImg = cv2.resize(img,(int(w/20),int(h/20)))#这里是w，h
                # cv2.imwrite(os.path.join(save_dir,"123.jpg"),reImg)
                # cv2.imshow(imgName,reImg)
                # cv2.waitKey(0)
                # img = img[:, :, (2, 1, 0)]
                fig, ax = plt.subplots(figsize=(12, 12))
                ax.imshow(img, aspect='equal')
                print imgName
                # cv2.imshow("img",img)
                # cv2.waitKey(0)
            for chil in childs.iter('object'):
                for name in chil.iter('name'):
                    class_name = name.text
                    print class_name
                for xmin,ymin,xmax,ymax in chil.iter('bndbox'):
                    print xmin.text,ymin.text,xmax.text,ymax.text
                ax.add_patch(plt.Rectangle((int(xmin.text), int(ymin.text)), int(xmax.text) - int(xmin.text), int(ymax.text) - int(ymin.text), fill=False,
                                           edgecolor='blue', linewidth=2))
                ax.text(int(xmin.text), int(ymin.text) - 2, '{:s}'.format(class_name), bbox=dict(facecolor='m', alpha=0.5),
                        fontsize=14, color='white')
                plt.axis('off')
                plt.tight_layout()
                plt.draw()
                output_dir = os.path.join(save_dir, imgName.split('.')[0] + "_detect_rst.jpg")
                plt.savefig(output_dir)


