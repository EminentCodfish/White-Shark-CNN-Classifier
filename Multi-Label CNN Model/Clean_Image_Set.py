# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 17:23:27 2020

@author: Deep Thought
"""
from iptcinfo3 import IPTCInfo
import cv2, os

path = "D:\\Population Study Videos\\Training Data\\CNN_training_set_2017\\"
kw_filter = ['Shark', 'No_shark', 'Gill', 'Pelvic', 'Caudal', 'Dorsal']
count = 1
#Filewalker function to iterate through the image set
for root, dirs, filenames in os.walk(path):
    for f in filenames:
        if f[-4:] == '.jpg': #check to see if the file extension is correct
            if f[4:] == 'gopr':
                pass
            if f[4:] == 'GOPR':
                pass
            else:
                #Open the file with IPTC
                info = IPTCInfo(root + '\\' +f)
                #print(info)
                meta = info['keywords'] #Extracts metadata
                meta_str = '' #Create a metadata string 
                meta_list = [] #Create a formatted list of image tags
                #print(f, meta)
                
                #iterate through the image tags to create a metadata tag string
                for i in range(len(meta)):
                    if meta[i].decode('ascii') in kw_filter:
                        meta_list.append(meta[i].decode('ascii'))
                        meta_str = meta_str + ' ' + meta[i].decode('ascii')
                        
                #There were some 'Shark' images with no meta-data.
                if 'Shark' not in meta_list:
                    if 'No_shark' not in meta_list:
                        meta_str = meta_str + ' ' + 'Shark'
                
                print(f, meta_str, count)
                count = count+1
    
                img = cv2.imread(root + '\\' +f)
                cv2.putText(img = img, text = f, org = (50,50), fontFace = cv2.FONT_HERSHEY_PLAIN, fontScale = 2, color = (255,255,255))
                cv2.putText(img = img, text = meta_str, org = (50,100), fontFace = cv2.FONT_HERSHEY_PLAIN, fontScale = 2, color = (255,255,255))
                img = cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5)
                cv2.imshow("Shark", img)
                k = cv2.waitKey(0)
                
                if k == 27:
                    pass
                
 