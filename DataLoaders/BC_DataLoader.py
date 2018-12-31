# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 11:09:31 2018

@author: sage
"""

from DataLoaders.TwoD_DataLoader import TwoD_DataLoader

from DataLoaders.RN_DataLoader import load_annotations
from DataLoaders.DataPoint import DataPoint

from DataUtils.tools import determine_crop

from config import config
import csv

def get_leak_ranges(rows):
    '''Method specifically designed for loading slice ranges for Endoleaks'''
    
    leak_ranges = []
    
    for line in rows:
        if line[0] != '':
            leak_ranges.append([int(line[0], int(line[1]))])
            
            if line[2] != '':
                leak_ranges.append([int(line[2], int(line[3]))])
                
                if line[4] != '':
                    leak_ranges.append([int(line[4], int(line[5]))])
                    
    return leak_ranges
    
    
class BC_DataLoader(TwoD_DataLoader):
    '''DataLoader class for binary classification, inherits from TwoD_DataLoader,
       as this is for 2D binary classification...'''
    
    def __init__(self,
                 init_location,   
                 label_location,
                 annotations,
                 ):
        
        '''
        Either annotations or annotations_loc must be provided.
        
        init_location - the location of the raw data
        label location - the location of the binary labels
        annotations - a list of data points with just name, slice and label,
                      or the location of a file with ^
        '''
        
        super().__init__(init_location, label_location)
        

        if type(annotations) == str:
            self.annotations = load_annotations(annotations)
        else:
            self.annotations = annotations
        
    def load_labels(self):
        '''Overrides load labels function from RN_DataLoader -
           loads in label information around which slices should be labelled
           1, and which are 0, the big Ohhh.'''
        
        self.file_names = set()
        self.data_points = []
        
        with open(self.label_location) as csvfile:
            reader = csv.reader(csvfile)
    
            for row in reader:
                
                name = row[0]
                self.file_names.add(name)
                
                slc_range = [int(row[1]), int(row[2])]
                label_ranges = get_leak_ranges(row[4:])
                
                for i in range(slc_range[0], slc_range[1]+1):
                    
                    label = 0
                    slc = i
                    
                    #Check if current slice is within a label range for positive
                    for interval in label_ranges:
                        if i >= interval[0] and i <= interval[1]:
                            label = 1
                            
                    self.data_points.append(DataPoint(name=name,
                                            label=label,
                                            in_memory=config['in_memory'],
                                            memory_loc=config['memory_loc'],
                                            slc=slc))
        
        
    def get_ann_from_ind(self, indx):
        ''' find the relevant label from an index for data points 
            in self.annotations. 
            
            indx - An index within self.data_points
        '''
            
        ref = self.data_points[indx].get_ref()
        ann_refs = [a.get_ref() for a in self.annotations]
        
        try:
            ind = ann_refs.index(ref)
            ann = self.annotations[ind].get_label(copy=True) #Why not
            
            ann = ann[:-1] #Remove class info, s.t. ann = [x0, y0, x1, y1]
            
            return ann
            
        except ValueError:
            print('Error - no annotation provided for ', ref)
        
    def initial_preprocess(self, image, indx):
        '''Initial preprocess first determines the correct crop, and then runs
           normal TwoD init preproc steps on the crop. 
           
           image - The image data to crop/ preprocess
           indx - The index of the image within self.data_points
        '''
        
        annotation = self.get_ann_from_ind(indx)
        new_shape = config['BC_input_size'][:-1] #No extra channel info added yet
        
        #Pads/Resizes the image to the desired dimensions
        image = determine_crop(annotation, image, new_shape, config['BC_pad_limit'])
        
        image = super().initial_preprocess(image)
