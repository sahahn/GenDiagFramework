#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 11:09:31 2018

@author: sage
"""
from DataLoaders.TwoD_DataLoader import TwoD_DataLoader
from config import config

import numpy as np
import csv

def get_name_slc(chunk):
    '''Specific function for loading retina-net style csv'''
    
    relevant_chunk = chunk.split('/')[-1].replace('.jpg','')
    name = relevant_chunk[:-3]
    slc = int(relevant_chunk[-3:])
    
    return name, slc

class RN_DataLoader(TwoD_DataLoader):
    
    def load_labels(self, include_none=True):
        
        self.file_names = set()
        self.data_points = []
        
        with open(self.label_location) as csvfile:
            reader = csv.reader(csvfile)
    
            for row in reader:
                name, slc = get_name_slc(row[0])
                self.file_names.add(name)
                
                try:
                    label = [float(row[i]) for i in range(1,5)]
                except ValueError:
                    label = [None]
                
                label.append(config['name_convs'][row[-1]])
                
                
                if label[0] == None and not include_none:
                    continue
                
                if label[0] == None:
                    label = np.empty((5))
                
                self.data_points.append(self.create_data_point(name, label, slc=slc))
                
                
def load_annotations(annotations_loc):
        '''Create an instance of the Retina Net DataLoader in order to load
           the data point w/ just label, name and slice information, and return
           the datapoints - notably loading only annotations with info'''
        
        RN_Loader = RN_DataLoader('fake/', annotations_loc)
        RN_Loader.load_labels(include_none=False)
        
        return RN_Loader.data_points
   