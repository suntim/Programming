# -*- coding: UTF-8 -*-
#!/usr/bin/env python

import os
import re
import shutil

save_dir = 'D:/newRailway/remain20171215'

img_list1 = []
img_list2 = []

img_dir1 = 'D:/newRailway/totall'
for fileName in os.listdir(img_dir1):
    if re.match(".*.jpg$",fileName):
        img_list1.append(fileName)
img_set1 = set(img_list1)

img_dir2 = 'D:/newRailway/subset'
for fileName in os.listdir(img_dir2):
    if re.match(".*.jpg$",fileName):
        img_list2.append(fileName)
img_set2 = set(img_list2)

# print img_set1
print "img_set1 len = %s"%len(img_set1)
# print img_set2
print "img_set2 len = %s"%len(img_set2)

intersection_img = img_set1 & img_set2
# print intersection_img
print "intersection len = %s"%len(intersection_img)

subtraction_img = img_set1-img_set2
# print subtraction_img
print "subtraction len = %s"%len(subtraction_img)

for imgName in subtraction_img:
    shutil.copy(os.path.join(img_dir1,imgName),os.path.join(save_dir,imgName))
print "Done!"
