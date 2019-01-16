# -*- coding: utf-8 -*-

from DataLoaders.DataLoader import DataLoader
from DataUtils.tools import normalize_data, resample
import nibabel as nib
import numpy as np
import os


class IQ_DataLoader(DataLoader):
    
    def __init__(self,
                 init_location,    
                 label_location,
                 seg_input_size=(256,256,256,1),
                 limit=None,
                 in_memory=True,
                 memory_loc=None,
                 compress=False,
                 preloaded=False
                 ):
        
         super().__init__(init_location, label_location, in_memory, memory_loc,
                         compress, preloaded)
         
         self.seg_input_size = seg_input_size
         
         if limit == None:
             self.limit = 10000000
         else:
             self.limit = limit
    
    def load_labels(self):
        
        self.iq_dict = {}
        
        ids = []
        scores = []
        
        with open(self.label_location, 'r') as f:
            lines = f.readlines()
            
            for line in lines:
                if 'NDAR' in line:
                    line = line.split(',')
                    
                    ids.append(line[0])
                    scores.append(float(line[1].strip()))
                    
                    
                    
        scores = normalize_data(np.array(scores))
        
        for i in range(len(ids)):
            self.iq_dict[ids[i]] = scores[i]
        
        
    def load_data(self):
        
        names = os.listdir(self.init_location)
        names = [name for name in names if 'NDAR' in name]
        
        for name in names:
            if len(self.data_points) < self.limit:
            
                label = self.iq_dict[name]
                dp = self.create_data_point(name, label)
                
                raw_file = nib.load(self.init_location + name + '/baseline/structural/t1_brain.nii.gz')
                dp.set_affine(raw_file.affine)
                
                data = raw_file.get_data() 
                data = normalize_data(data)
                data = np.expand_dims(data, axis=-1)
                
                shp = np.shape(data)
                
                if shp < self.seg_input_size:
                    new_data = np.zeros(self.seg_input_size)
                    
                    dif1 = int(np.floor(np.shape(new_data)[0] - np.shape(data)[0]))
                    dif2 = int(np.ceil(np.shape(new_data)[0] - np.shape(data)[0]))
                    
                    new_data[dif1:shp[0]+dif2, dif1:shp[1]+dif2, dif1:shp[2]+dif2] = data
                    
                elif shp != self.seg_input_size:
                    new_data = resample(data, self.seg_input_size)
                    
                else:
                    new_data = data
                    
                dp.set_data(new_data)
                
            
            
#/baseline/structural/t1_brain.nii.gz or t1_gm_parc.nii.gz
        
        
