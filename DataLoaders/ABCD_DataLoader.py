# -*- coding: utf-8 -*-

from DataLoaders.DataLoader import DataLoader
from DataUtils.tools import standardize_data, normalize_data, resample
from DataUtils.crop_tools import get_crop_ind, fill_to
from DataUtils.loader_helper import smart_load, read_t_transform
from nilearn.image import resample_img
import nibabel as nib
import nilearn
import numpy as np
import os


class ABCD_DataLoader(DataLoader):
    
    def __init__(self,
                 init_location,    
                 label_location,
                 label_key='NDAR',
                 file_key='brain.finalsurfs.mgz',
                 input_size=(256,256,256,1),
                 load_segs=False,
                 segs_key='aparc.a2009s+aseg.mgz',
                 tal_transform=True,
                 tal_key='talairach.xfm',
                 limit=None,
                 in_memory=True,
                 memory_loc=None,
                 compress=False,
                 preloaded=False
                 ):
        
         super().__init__(init_location, label_location, in_memory, memory_loc,
                         compress, preloaded)
         
         self.label_key = label_key
         self.file_key = file_key
         self.input_size = input_size
         self.load_segs = load_segs
         self.segs_key = segs_key
         self.tal_transform = tal_transform
         self.tal_key = tal_key
         
         if limit == None:
             self.limit = 10000000
         else:
             self.limit = limit
    
    def load_labels(self):
        
        self.label_dict = {}
        
        with open(self.label_location, 'r') as f:
            lines = f.readlines()
            
            for line in lines:
                if self.label_key in line:
                    line = line.split(',')
                    self.label_dict[line[0]] = float(line[1].strip())
            
    def load_data(self):

        shapes = []
        
        file_names = os.listdir(self.init_location)
        names = [name for name in file_names if name in self.label_dict]
        
        for name in names:
            if (len(self.data_points) < self.limit):
                dp = self.create_data_point(name, self.label_dict[name])
                
                if self.preloaded == False:

                    try:
                        path = os.path.join(self.init_location, name, self.file_key)
                        raw_file = smart_load(path)

                        if self.tal_transform:
                            tal_affine = read_t_transform(os.path.join(self.init_location, name, self.tal_key))
                            new_affine = raw_file.affine.dot(tal_affine)
                            raw_file   = resample_img(raw_file, target_affine=new_affine, interpolation="continuous")

                        dp.set_affine(raw_file.affine)
                        data = raw_file.get_fdata()
                        data = normalize_data(data)

                        xs, ys = get_crop_ind(data)
                        data = data[xs[0]:ys[0], xs[1]:ys[1], xs[2]:ys[2]]

                        shapes.append(np.shape(data))

                        data = np.expand_dims(data, axis=-1)
                        data = fill_to(data, self.input_size)


                        if self.load_segs:
                            
                            seg_path = os.path.join(self.init_location, name, self.segs_key)
                            raw_seg = smart_load(seg_path)

                            if self.tal_transform:
                                tal_affine = read_t_transform(os.path.join(self.init_location, name, self.tal_key))
                                new_affine = raw_seg.affine.dot(tal_affine)
                                raw_seg   = resample_img(raw_seg, target_affine=new_affine, interpolation="nearest")
                                
                            seg = raw_seg.get_data()
                            seg = seg[xs[0]:ys[0], xs[1]:ys[1], xs[2]:ys[2]]

                            seg = np.expand_dims(seg, axis=-1)
                            seg = fill_to(seg, self.input_size)

                        if np.shape(data) != self.input_size:
                            print('resample', np.shape(data), name)
                            data = resample(data, self.input_size)

                            if self.load_segs:
                                seg = resample(seg, self.input_size)
            
                        dp.set_data(data)

                        if self.load_segs:
                            dp.set_guide_label(seg)
                    except:
                        print('error with', name)

                self.data_points.append(dp)

        print(np.max(shapes, axis=0))

    #All Unique patients, so just override get_patient, w/ get name instead
    def get_unique_patients(self):
        
        patients = sorted([dp.get_name() for dp in self.data_points])
        return np.array(patients)
        
        
