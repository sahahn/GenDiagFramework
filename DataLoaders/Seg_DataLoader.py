# -*- coding: utf-8 -*-

from DataLoaders.DataLoader import DataLoader
from DataLoaders.RN_DataLoader import load_annotations
from DataUtils.tools import determine_crop_3d, calculate_ranges, normalize_data

from DataLoaders.DataPoint import DataPoint
from config import config
import os
import numpy as np
import nibabel as nib


class Seg_DataLoader(DataLoader):
    '''Base class for 3D segmentation loader,
       where label_location refers to a folder containing segmentations.'''

    
    def __init__(self,
                 init_location,
                 label_location, 
                 annotations,
                 label_type='full',
                 seg_key='endo',
                 pad_info=(0,7)
                 ):
        ''' 
        annotations - a list of data points with just name, slice and label,
                      or the location of a file with ^
        label_type - refers to if the underlying segmentation is saved
                     on the full scan size, or if label_type = 'crop'
                     then segmentations are saved with specific crop details
        seg_key - e.g. 'filename_endo', 'filename_seg', 'filename_full',
                  so the filename followed by_ and the seg_key
        pad_info - (base_pad, axial_pad), if label_type = 'crop', then
                     ensure that the pad_info matches the saved format,
                     **important**
        '''
        
        super().__init__(init_location, label_location)
        
        if self.label_location[-1] != '/':
             self.label_location += '/'
        
        self.label_type = label_type
        self.pad_info = pad_info
        self.seg_key = seg_key
        
        #If a loaction, load annotations from location, otherwise
        #assume annotations is a list of datapoint and load ranges.
        if type(annotations) == str:
            anns = load_annotations(annotations)
        else:
            anns = annotations
            
        self.range_dict = calculate_ranges(anns)
            
    def load_labels(self):
        
        self.data_points = []
        
        seg_files = [file for file in os.listdir(self.label_location) if
                     self.seg_key in file]
        
        for file in seg_files:
        
            #Remove file extension + seg_key
            name = file.split('.')[0].replace('_' + self.seg_key, '') 
            
            seg_path = self.label_location + file
            nib_file = nib.load(seg_path)

            #If the seg_label is for the full image, it will be resized when
            #loading the data
            label = nib_file.get_data()
  
            self.data_points.append(DataPoint(name=name, label=label,
                                              in_memory=config['in_memory'],
                                              memory_loc=config['memory_loc']))
        
    def load_data(self):
        
        for i in range(len(self.data_points)):
            
            name = self.data_points[i].name
            raw_file_path = self.init_location + name + '.nii'
            
            try:
                raw_file = nib.load(raw_file_path)
            except:
                raw_file = nib.load(raw_file_path + '.gz')
            
            self.data_points[i].set_pixdims(raw_file.header['pixdim'])
            self.data_points[i].set_affine(raw_file.affine)
            
            thickness = self.data_points[i].get_thickness()
            new_shape = config['Seg_input_size'][1:]  #Channels first
            
            data = raw_file.get_data()
            ranges = self.range_dict[name]
            
            
            data = determine_crop_3d(data, ranges, thickness,
                       new_shape, self.pad_info[0], self.pad_info[1])
            
            
            data = np.clip(data, *config['clip_range'])
            data = normalize_data(data)
            data = np.expand_dims(data, axis=0) #Channels first by default 
            
            self.data_points[i].set_data(data)
            
            if self.label_type == 'full':
                    self.data_points[i].label = determine_crop_3d( 
                            self.data_points[i].label, ranges, thickness,
                            new_shape, self.pad_info[0], self.pad_info[1])
            
            #Channels first by default
            self.data_points[i].label = np.expand_dims(self.data_points[i].label, axis=0)





