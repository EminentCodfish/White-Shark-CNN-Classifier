# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 12:43:55 2020

@author: Chris Rillahan

This is a collection of functions which create a single image set with labels for 
use in training a CNN.  
"""
import random, os
from shutil import copyfile
from iptcinfo3 import IPTCInfo

def ImageSubSample(inputPath, outputPath, SampleProp):
    #This function creates a image subset of images from a image set.
    #The size of the subset is derived from the SampleProp variable which 
    #takes a proportion of the total image set.
    
    #Filewalker function to iterate through images
    for root, dirs, filenames in os.walk(inputPath):
        for f in filenames:
            if f[-5:] == '.jpeg':
                if random.random() < SampleProp:             
                    copyfile(root +'\\' + f, outputPath + f)
            if f[-4:] == '.jpg':
                if random.random() < SampleProp:             
                    copyfile(root +'\\' + f, outputPath + f)
                    
def ImageAgg(inputPath, outputPath):
    #This function collates and copies images from directory into a single folder.
    
    #Filewalker function
    for root, dirs, filenames in os.walk(inputPath):
        for f in filenames:
            copyfile(root +'\\' + f, outputPath + f)
            
def imageMetaLabel(path, metaFileName, kw_filter):
    #This function creates a csv listing the file name and associated image tags.
    #The image tags are selected from a list of known tags listed in kw_filter.

    ##Open file
    data = open(path + metaFileName, 'w')
    data.write('Name,Label' + '\n') 

    #Filewalker function to iterate through the image set
    for root, dirs, filenames in os.walk(path):
        for f in filenames:
            if f[-4:] == '.jpg': #check to see if the file extension is correct
                
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
                 
                #Write the image tag metadata to file    
                data.write(str(f) + ',' + meta_str + '\n')   
                
            if f[-5:] == '.jpeg': #Do the same with *.jpeg files
                info = IPTCInfo(root + '\\' +f)
                meta = info['keywords']
                meta_str = ''
                meta_list = []
                #print(f, meta)
                
                for i in range(len(meta)):
                    if meta[i].decode('ascii') in kw_filter:
                        meta_list.append(meta[i].decode('ascii'))
                        meta_str = meta_str + ' ' + meta[i].decode('ascii')
                        
                if 'Shark' not in meta_list:
                    if 'No_shark' not in meta_list:
                        meta_str = meta_str + ' ' + 'Shark'
                        
                data.write(str(f) + ',' + meta_str + '\n') 


