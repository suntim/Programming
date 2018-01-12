# -*- coding: UTF-8 -*-
#!/usr/bin/env python
import os
import re
import shutil

img_dir = unicode(r'C:\Users\x\Desktop\TRAIN','utf-8')
xml_dir = unicode(r"C:\Users\x\Desktop\label",'utf-8')
save_dir = unicode(r"C:\Users\x\Desktop\TRAIN",'utf-8')

for fileName in os.listdir(img_dir):
    if re.match(".*.jpg$",fileName):
        img_path = os.path.join(img_dir, fileName.split(".")[0]+".jpg")
        save_img_path = os.path.join(save_dir, fileName.split(".")[0]+".jpg")
        xml_path = os.path.join(xml_dir, fileName.split(".")[0]+".xml")
        save_xml_path = os.path.join(save_dir, fileName.split(".")[0]+".xml")
        # print save_img_path
        # shutil.copy(img_path,save_img_path)
        print save_xml_path
        shutil.copy(xml_path,save_xml_path)
