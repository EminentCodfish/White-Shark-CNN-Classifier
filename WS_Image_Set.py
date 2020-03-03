# -*- coding: utf-8 -*-
"""
This script creates a image dataset from the Fast.AI multi-label CNN classifier.
This dataset is a collection from existing images.  Currently we have ~7,500 
shark images which have detailed labels.  There is an additional ~110,000 shark
images with minimal labeling, some simply labeled as a shark image.  Finally, 
there are ~530,000 images labeled 'No Shark'.  This scripts attempts to maximize
the use of the images with detailed labels while also using other available image
without creating a very uneven dataset.

Created on Thu Jan  9 15:46:28 2020

@author: Deep Thought
"""
                
import WS_Utils as Util

'''No Shark image set subset'''
#Subset a series of images simply labeled 'No Shark'.  The total data set is 
#currently 533,956 images.

#Proportion of 'No Shark' images .  This should yield ~16,000 images.
NS_Prop = 0.05

#Image path
path = "D:\\Population Study Videos\\Training Data\\Training Images\\No shark\\"

#subset path
output_path = "D:\\Population Study Videos\\Training Data\\CNN_training_set\\"

Util.ImageSubSample(path, output_path, NS_Prop)


'''Shark subset image set'''
#Subset a series of images simply labeled 'Shark'.  The total data set is 
#currently 109,674 images.

#Proportion of 'Shark' images.  This should yield ~11,000 images 
NS = 0.1

#Image path
path = "D:\\Population Study Videos\\Training Data\\Training Images\\Shark\\"

Util.ImageSubSample(path, output_path, NS)


'''Collate the labeled Shark image set'''
#Shark images with detailed labels are currently in multiple directories and
#sub-directories.  This function moves them all into the training folder.

#Image path
path = "D:\\Population Study Videos\\Training Data\\Training Images\\ID Catalog\\"

Util.ImageAgg(path, output_path)


'''Meta-data labeling'''
#The FastAI multi-label CNN requires a csv listing the filename and labels. 

#Image path
path = "D:\\Population Study Videos\\Training Data\\CNN_training_set\\"

metaFileName = 'ws_metadata.csv'

#keyword_filter
kw_filter = ['Shark', 'No_shark', 'Gill', 'Pelvic', 'Caudal', 'Dorsal']

Util.imageMetaLabel(path, metaFileName, kw_filter)
