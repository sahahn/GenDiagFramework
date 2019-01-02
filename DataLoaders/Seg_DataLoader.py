# -*- coding: utf-8 -*-

from DataLoaders.DataLoader import DataLoader
from DataLoaders.RN_DataLoader import load_annotations
from DataUtils.tools import determine_crop_3d, calculate_ranges, normalize_data

import os
from config import config
import numpy as np
import nibabel as nib


class Seg_DataLoader(DataLoader):
    '''Base class for 3D segmentation loader '''
    
    def __init__(self,
                 init_location,
                 label_location, 
                 annotations,
                 label_type='full',
                 seg_key='endo',
                 n_classes=1,
                 pad_info=(0,7),
                 in_memory=True,
                 memory_loc=None,
                 compress=False,
                 preloaded=False
                 ):
        ''' 
        label_location - A folder containing the segmentations to read in.
        annotations - a list of data points with just name, slice and label,
                      or the location of a file with ^
        label_type - refers to if the underlying segmentation is saved
                     on the full scan size, or if label_type = 'crop'
                     then segmentations are saved with specific crop details.
        seg_key - e.g. 'filename_endo', 'filename_seg', 'filename_full',
                  so the filename followed by_ and the seg_key. If using
                  the child class Par_Seg_DataLoader then seg_key can refer to
                  a list of keys, where by default the first refers to the
                  partial key, and the second to any full segmentations. 
        n_classes -The the number of different classes with segmentations
                  must be provided, where it is assumed class 1 has label 1,
                  class2 label 2, ect... 
        pad_info - (base_pad, axial_pad), if label_type = 'crop', then
                     ensure that the pad_info matches the saved format,
                     **important**
        '''
        
        super().__init__(init_location, label_location, in_memory, memory_loc,
                         compress, preloaded)
        
        if self.label_location[-1] != '/':
             self.label_location += '/'
        
        self.label_type = label_type
        self.pad_info = pad_info
        self.seg_key = seg_key
        self.n_classes = n_classes
        
        #If a location, load annotations from location, otherwise
        #assume annotations are a list of datapoint and load ranges.
        if type(annotations) == str:
            anns = load_annotations(annotations)
        else:
            anns = annotations
            
        self.range_dict = calculate_ranges(anns)
            
    def load_labels(self):
        
        seg_files = [file for file in os.listdir(self.label_location) if
                     self.seg_key in file]

        
        for file in seg_files:
        
            name = file.split('.')[0].replace('_' + self.seg_key, '') 
            nib_file = nib.load(self.label_location + file)

            #If the seg_label is for the full image, it will be resized when loading data
            label = nib_file.get_data()
            
            if self.n_classes > 1:
                label = np.array(self.load_multiclass_seg(label))
                
            self.data_points.append(self.create_data_point(name, label))
            
    def load_data(self):
        
        for i in range(len(self.data_points)):
            
            name = self.data_points[i].name
            raw_file_path = self.init_location + name + '.nii'
            
            try:
                raw_file = nib.load(raw_file_path)
            except:
                raw_file = nib.load(raw_file_path + '.gz')
            
            self.data_points[i].set_pixdims(raw_file.header['pixdim'])
            
            thickness = self.data_points[i].get_thickness()
            new_shape = config['Seg_input_size'][1:]  #Channels first
            
            data = raw_file.get_data()
            ranges = self.range_dict[name]
            affine = raw_file.affine
            
            data, new_affine = determine_crop_3d(data, ranges, thickness,
                       new_shape, self.pad_info[0], self.pad_info[1], affine)

            data = np.clip(data, *config['clip_range'])
            data = normalize_data(data)
            data = np.expand_dims(data, axis=0) #Channels first by default 
            
            self.data_points[i].set_data(data)
            self.data_points[i].set_affine(new_affine)
            
            #If label type is full, must resize labels
            if self.label_type == 'full':
                label =  self.data_points[i].get_label()
                
                #If one channel, must add by default new channel first
                if self.n_classes == 1:
                
                    new_label = determine_crop_3d( 
                        label, ranges, thickness, new_shape, self.pad_info[0],
                        self.pad_info[1], affine)[0] #[0] select justs label
               
                    self.data_points[i].set_label(np.expand_dims(
                                                new_label, axis=0))
        
                #Otherwise if multiclass, must recrop each channel
                else:
                    
                    new_label = [ determine_crop_3d(channel, ranges, thickness,
                                new_shape, self.pad_info[0], self.pad_info[1],
                                affine)[0] for channel in label ]
            
                    self.data_points[i].set_label(np.array(new_label))
                
            
    def load_multiclass_seg(self, raw_label):
        '''From a raw segmentation with multiple classes, return a
           list containing seperate segmentation channel converted to 1.
           Note: this assumes segmentations are labeled 1 for class 1, ect..'''
      
        label = []
        
        for i in range(1, self.n_classes+1):
            seg = np.copy(raw_label)
            
            seg[seg!=i] = 0
            seg[seg==i] = 1
            
            label.append(seg)
            
        return label
        
    
        
    
            





