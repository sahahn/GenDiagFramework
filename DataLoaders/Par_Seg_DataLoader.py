# -*- coding: utf-8 -*-

from DataLoaders.Seg_DataLoader import Seg_DataLoader
from DataUtils.Seg_tools import get_seen

import os
import numpy as np
import nibabel as nib

class Par_Seg_DataLoader(Seg_DataLoader):
    '''Overrides Seg_DataLoader to provide support for loading partial
       segmentations. Note: the same constructor for Seg_DataLoader is used,
       though seg_key must be passed a list of 2 keys instead of one, where the
       first is the partial keyword and the second is for fulls'''
   
    def load_labels(self):
        
        self.data_points = []
        
        seg_files = [file for file in os.listdir(self.label_location) if
                     self.seg_key[0] in file or self.seg_key[1] in file]
        
        for file in seg_files:
            
            #Use 'key' as quick flag for if this file is partial or full,
            #By default key == 0 is partial, key == 1 is full
            key = 0
            if self.seg_key[1] in file:
                key = 1
            
            name = file.split('.')[0].replace('_' + self.seg_key[key], '') 
            nib_file = nib.load(self.label_location + file)
            
            label = nib_file.get_data()
            seen = self.load_seen_label(label, key)
            
            if self.n_classes > 1:
                label = self.load_multiclass_seg(label)
            else:
                label = [label]
                
            label.append(seen)
            label = np.array(label)
  
            self.data_points.append(self.create_data_point(name, label))
            
        #Because of the way load_data is setup, and the addition of the seen
        #Channel. Set n_classes+=1, s.t. expand dims won't be called, once done
        #with loading the labels, and the seen label can be resized if needed
        self.n_classes+=1
    
    def load_seen_label(self, raw_label, key):
        '''Input is a 'raw label', or in other words a partial segmentation,
           and output is the seen label, here it is assumed
           the start/axial partial seen flag is n_classes+1'''

        if key == 0:
            seen = get_seen(raw_label, self.n_classes+1)
         
        else:
            seen = np.ones(np.shape(raw_label))
        
        return seen.astype('float')
      
    
    
    