# -*- coding: utf-8 -*-

from DataLoaders.DataLoader import DataLoader
from DataLoaders.RN_DataLoader import load_annotations
from DataUtils.tools import determine_crop_3d, calculate_ranges, normalize_data
from DataUtils.Seg_tools import get_seen

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
                 n_classes=1,
                 pad_info=(0,7)
                 ):
        ''' 
        annotations - a list of data points with just name, slice and label,
                      or the location of a file with ^
        label_type - refers to if the underlying segmentation is saved
                     on the full scan size, or if label_type = 'crop'
                     then segmentations are saved with specific crop details
        seg_key - e.g. 'filename_endo', 'filename_seg', 'filename_full',
                  so the filename followed by_ and the seg_key. If label_type
                  is set to 'partial', then seg_key can refer to a list of keys,
                  where by defaults the first refers to the partial key, and the
                  second to any full segmentations.
        n_classes - If label_type is partial, then the number of different classes
                  must be provided (where it is assumed class 1 has label 1, class
                  2 label 2, ect... in calculating partial segs)
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
        self.n_classes = n_classes
        
        if type(seg_key) != list and self.label_type == 'partial':
            print('Must provide a list of seg keys when loading partials')
        
        #If a location, load annotations from location, otherwise
        #assume annotations are a list of datapoint and load ranges.
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
            
        ### IF PARTIAL SEG! ###
        '''
        raw_label = some_load_seg_function
       
        #Where it is assumed the start/axial partial seen flag is n_classes+1
        seen = get_seen(raw_label, self.n_classes+1)
        seen = seen.astype('float')
        
        label = []
        
        for i in range(1, self.n_classes+1):
            seg = np.copy(raw_label)
            
            seg[seg!=i] = 0
            seg[seg==i] = 1
            
            label.append(seg)
            
        label.append(seen)
        label = np.array(label)'''
            
        
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
            
            if self.label_type == 'full':
                new_label, new_affine = determine_crop_3d( 
                    self.data_points[i].label, ranges, thickness,
                    new_shape, self.pad_info[0], self.pad_info[1], affine)
                
                #Channels first by default - only need to do this for full labels
                self.data_points[i].label = np.expand_dims(new_label, axis=0)
            
            
            
    
        
    
            





