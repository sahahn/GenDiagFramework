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
import nibabel.processing

class ABCD_DataLoader(DataLoader):
    
    def __init__(self,
                 init_location,    
                 label_location,
                 label_key='NDAR',
                 load_extra_info=False,
                 file_key='brain.finalsurfs.mgz',
                 input_size=(256,256,256,1),
                 load_segs=False,
                 segs_key='aparc.a2009s+aseg.mgz',
                 min_max=None,
                 limit=None,
                 in_memory=True,
                 memory_loc=None,
                 compress=False,
                 preloaded=False
                 ):
        
        super().__init__(init_location, label_location, in_memory, memory_loc,
                         compress, preloaded)
         
        self.label_key = label_key
        self.load_extra_info = load_extra_info
        self.file_key = file_key
        self.input_size = input_size
        self.load_segs = load_segs
        self.segs_key = segs_key
        self.min_max = min_max

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

                    if self.load_extra_info:
                        self.label_dict[line[0]] = [float(l.stip()) for l in line[1:]]
                    else:
                        self.label_dict[line[0]] = float(line[1].strip())
            
    def load_data(self):

        shapes = []
        
        file_names = os.listdir(self.init_location)
        names = [name for name in file_names if name in self.label_dict]
        
        for name in names:
            if (len(self.data_points) < self.limit):
                dp = self.create_data_point(name, self.label_dict[name])
                
                if self.preloaded == False:

                    path = os.path.join(self.init_location, name, self.file_key)
                    raw_file = smart_load(path)
                    
                    dp.set_affine(raw_file.affine)
                    data = raw_file.get_fdata()
                    data = normalize_data(data, self.min_max)

                    xs, ys = get_crop_ind(data)
                    data = data[xs[0]:ys[0], xs[1]:ys[1], xs[2]:ys[2]]

                    shapes.append(np.shape(data))

                    data = np.expand_dims(data, axis=-1)
                    data = fill_to(data, self.input_size)

                    if self.load_segs:

                        try:
                            seg_path = os.path.join(self.init_location, name, self.segs_key)
                            raw_seg = smart_load(seg_path)

                            seg = raw_seg.get_data()
                            seg = seg[xs[0]:ys[0], xs[1]:ys[1], xs[2]:ys[2]]

                            seg = np.expand_dims(seg, axis=-1)
                            seg = fill_to(seg, self.input_size)
                        except:
                            print('error loading segmentation ', )

                    if np.shape(data) != self.input_size:
                        
                        print('resample', np.shape(data), name)
                        data = resample(data, self.input_size)

                        if self.load_segs:
                            seg = resample(seg, self.input_size)
        
                    dp.set_data(data)

                    if self.load_segs:
                        dp.set_guide_label(seg)

                self.data_points.append(dp)
        
        if len(shapes) > 0:
            print(np.max(shapes, axis=0))

        print(len(self.data_points), 'loaded')

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


    def get_splits_by_site(self, test_sites=[], val_sites=[]):
        '''Provide input for splits as list/set of sites to include in test set only or validation set only'''

        if test_sites == None:
            test_sites = []
        if val_sites == None:
            val_sites = []

        self.load_all()
        train, test, val = [], [], []

        for dp in self.data_points:
            site = int(dp.get_extra()[0])

            if site not in test_sites and site not in val_sites:
                train.append(dp)
            elif site in test_sites:
                test.append(dp)
            elif site in val_sites:
                val.append(dp)

        if len(val) == 0 and len(test) == 0:
            return train
        elif len(val) == 0:
            return train, test
        elif len(test) == 0:
            return train, val
        else:
            return train, test, val



        
        

