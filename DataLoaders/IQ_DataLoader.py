# -*- coding: utf-8 -*-

from DataLoaders.DataLoader import DataLoader
from DataUtils.tools import standardize_data, normalize_data, reverse_standardize_data, resample
import nibabel as nib
import nilearn
import numpy as np
import os


class IQ_DataLoader(DataLoader):
    '''Dataloader for loading structural T1 MRIs, originally used for predicted iq, thus
       IQ_DataLoader, but really general purpose. '''
    
    def __init__(self,
                 init_location,    
                 label_location,
                 input_size=(256,256,256,1),
                 limit=None,
                 scale_labels=False,
                 in_memory=True,
                 memory_loc=None,
                 compress=False,
                 preloaded=False
                 ):
        
         super().__init__(init_location, label_location, in_memory, memory_loc,
                         compress, preloaded)
         
         self.input_size = input_size
         self.scale_labels = scale_labels
         self.scale_info = None
         
         if limit == None:
             self.limit = 10000000
         else:
             self.limit = limit
    
    def load_labels(self):
        '''Will work to load any labels of the form, filename w/ NDAR and then a score'''
        
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
                    
        
        if self.scale_labels:
            '''Dont know if this is ever a good idea '''
            scores, self.scale_info = standardize_data(np.array(scores),
                                                     return_reverse=True)

        for i in range(len(ids)):
            self.iq_dict[ids[i]] = scores[i]
            
        
        
    def load_data(self):
        
        names = os.listdir(self.init_location)
        names = [name for name in names if 'NDAR' in name]
        
        for name in names:
            if (len(self.data_points) < self.limit) and (name in self.iq_dict):
            
                label = self.iq_dict[name]
                dp = self.create_data_point(name, label)
                
                if self.preloaded == False:
                    try:
                        #raw_file = nib.load(self.init_location + name + '/baseline/structural/t1_brain.nii.gz')
                        raw_file = nilearn.image.load_img(self.init_location + name + '/baseline/structural/t1_brain.nii.gz')
                        raw_file = nilearn.image.crop_img(raw_file)
                    
                    except FileNotFoundError:
                        print('missing: ', name)
                        continue
                
                    dp.set_affine(raw_file.affine)
                
                    data = raw_file.get_data() 
                    data = standardize_data(data)
                    data = normalize_data(data)
                    data = np.expand_dims(data, axis=-1)
                
                    shp = np.shape(data)
                    
                    if (shp < np.array(self.input_size)).all():
                        new_data = np.zeros(self.input_size)
            
                        d1 = np.floor((np.shape(new_data) - np.shape(data))/2)
                        d2 = np.ceil((np.shape(new_data) - np.shape(data))/2)
                        new_data[d1[0]:shp[0]+d2[0], d1[1]:shp[1]+d2[1], d1[2]:shp[2]+d2[2]] = data
                    
                    elif shp != self.input_size:
                        new_data = resample(data, self.input_size)
                    
                    else:
                        new_data = data
                    
                    dp.set_data(new_data)
                
                self.data_points.append(dp)
                

    
    def reverse_label_scaling(self):
        
        for dp in self.data_points:
            label, pred_label = dp.get_label(), dp.get_pred_label()
            
            dp.set_label(reverse_standardize_data(label, self.scale_info))
            
            if pred_label != None:
                dp.set_pred_label(reverse_standardize_data(pred_label, self.scale_info))
    
    #All Unique patients, so just override get_patient, w/ get name instead
    def get_unique_patients(self):
        
        patients = sorted([dp.get_name() for dp in self.data_points])
        return np.array(patients)
    
    def get_data_points_by_patient(self, patients):
        
        relevant = []
        
        for dp in self.data_points:
            if dp.get_name() in patients:
                relevant.append(dp)
        
        return relevant
                
            
            
#/baseline/structural/t1_brain.nii.gz or t1_gm_parc.nii.gz
        
        
