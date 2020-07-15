# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 22:05:55 2020

@author: Deep Thought
"""

import pandas as pd

path = 'E:\\CNN_training_set_2017\\'

data = pd.read_csv(path + 'ws_metadata.csv')

shark = 0
only_shark = 0
no_shark = 0
gill = 0
dorsal = 0
pelvic = 0
caudal = 0

for i in range(len(data)):
    if 'Shark' in data.loc[i, 'Label']:
        shark = shark + 1
    if data.loc[i,'Label'] == ' Shark':
        only_shark = only_shark + 1
    if 'No_shark' in data.loc[i, 'Label']:
        no_shark = no_shark + 1
    if 'Gill' in data.loc[i, 'Label']:
        gill = gill + 1
    if 'Dorsal' in data.loc[i, 'Label']:
        dorsal = dorsal + 1
    if 'Pelvic' in data.loc[i, 'Label']:
        pelvic = pelvic + 1
    if 'Caudal' in data.loc[i, 'Label']:
        caudal = caudal + 1
        
print('No Shark: ', no_shark)        
print('Shark: ', shark)
print('Only Shark: ', only_shark)
print('Gill: ', gill)
print('Dorsal: ', dorsal)
print('Pelvic: ', pelvic)
print('Caudal: ', caudal)
