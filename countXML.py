# -*- coding: UTF-8 -*-
#!/usr/bin/env python
import xml.etree.ElementTree as ET
import os
import re
import numpy as np


W_name = ["breakage","crack","waterdrop","spot"]
print len(W_name)
COUNT = np.zeros([1,len(W_name)],dtype=int)

with open('countXml.txt','w') as f:
    xml_dir = unicode(r'D:\Glass\Result\Test\TEST','utf-8')
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
