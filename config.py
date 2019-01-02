#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 12:08:59 2018

@author: sage
"""
import os

config = {}



#Could also hypothetically only load some data into memory, some load~
config['in_memory'] = False
config['memory_loc'] = '/mnt/sda5/temp/'
config['compress'] = True


#Where to save models
config['model_loc'] = '/home/sage/GenDiagFramework/saved_models/'


# Retina Net (RN) Config Information #
######################################

config['name_convs'] = {'AAA': 0, '': None}
config['RN_input_size'] = (512, 512, 1)


# Binary Classification Config Information #
############################################

#Input size to the network
config['BC_input_size'] = (128, 128, 1)

#Max that each dim. can be padded before reshape- based on
#the given crop labels. Set to 0 for just reshape. 
config['BC_pad_limit'] = 5 


# 3D Segmentation Config Information #
############################################

#Input size for 3D segmentation network
config['Seg_input_size'] = (1, 128, 128, 128)





#Global clip range - if none, set [-big number, big number]
config['clip_range'] = [-100, 600]



#For all location configs, make sure they are valid directories, if not make one
locs = [config['memory_loc'], config['model_loc']]
    
for l in locs:
    if not os.path.exists(l):
        os.makedirs(l)
        

        
    
