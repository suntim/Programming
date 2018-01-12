# -*- coding: UTF-8 -*-
#!/usr/bin/env python
import os
import sys
ROOT_P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_P)
import re

img_dir = unicode(r'D:\Glass\Result\Test\test1','utf-8')
with open("test.txt",'w') as f:
    for fileName in os.listdir(img_dir):
        if re.match(".*.xml$", fileName):
            f.write(fileName.split('.')[0]+'\n')
    print ("Done!")
