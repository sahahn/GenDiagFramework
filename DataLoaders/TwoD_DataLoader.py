
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 11:09:31 2018

@author: sage
"""
from DataLoaders.DataLoader import DataLoader
from DataUtils.tools import standardize_data
from config import config

import numpy as np
import nibabel as nib

class TwoD_DataLoader(DataLoader):

    def load_data(self):
        
        indx_ref_list = [dp.get_ref() for dp in self.data_points]
        
        for name in self.file_names:
            
            raw_file_path = self.init_location + name + '.nii'
            
            try:
                raw_file = nib.load(raw_file_path)
            except:
                raw_file = nib.load(raw_file_path + '.gz')
            
            data = raw_file.get_data()
            
            #Sag. to axial conversion by default~
            #TODO make this a changable param
            data = data.transpose(2,1,0)
            
            for slc in range(len(data)):
                
                #Find placement within data_points
                try:
                    indx = indx_ref_list.index(name+str(slc))
                    
                    image = data[slc]
                    image = self.initial_preprocess(image, indx)
                    
                    self.data_points[indx].set_data(image)
                
                #Better to ask for forgiveness rather than permisission
                except ValueError:
                    pass
                    
            
    def initial_preprocess(self, image, indx):
        '''Takes in the image to preprocess, and the indx of the image in data
           points, if specific image proc. needs to be done.'''
        
        image = np.clip(image, *config['clip_range'])
        image = standardize_data(image)
        
        #Adds an extra channel
        image = np.expand_dims(image, axis=-1)
        
        #Uncomment to convert grayscale to 3 image copied grayscale
        #image = np.stack((image,)*3, axis=-1)
        
        return image
    
    
                