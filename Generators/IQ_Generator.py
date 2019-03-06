# -*- coding: utf-8 -*-

from Generators.Seg_Generator import Seg_Generator
import DataUtils.augment_3d as augment_3d
import numpy as np

class IQ_Generator(Seg_Generator):
    
    def __init__(self,
                 data_points,
                 dim,
                 batch_size,
                 n_classes,
                 shuffle,
                 augment,
                 label_size = None,
                 distort = False,
                 dist_scale = .1,
                 flip = False,
                 permute = False,
                 gauss_noise = 0,
                 rand_seg_remove = 0):
        
         super().__init__(data_points, dim, batch_size, n_classes, shuffle, augment,
              label_size, distort, dist_scale, flip, permute, gauss_noise)
         
         self.rand_seg_remove = rand_seg_remove
         
         
    def get_sample_from(self, ID):
        
        #If augment return a copy
        x = self.data_points[ID].get_data(copy=self.augment)
        y = self.data_points[ID].get_label(copy=self.augment)
        
        if self.augment:
            x,y = self.data_aug(x,
                                y,
                                self.data_points[ID].get_affine(copy=True),
                                self.data_points[ID].get_guide_label(copy=self.augment))
            
        return x, y
            
    def data_generation(self, list_IDs_temp):

        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, self.n_classes))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            X[i], y[i] = self.get_sample_from(ID)
       
        return X, y
        
    def data_aug(self, x, y, affine, seg):
        
        if self.distort:
            x = augment_3d.augment_just_data(x, affine,
                      flip=self.flip, scale_deviation=self.dist_scale)
            
        if self.permute:
            x = augment_3d.random_permutation_x(x)
            
        if self.gauss_noise != 0:
            x = augment_3d.add_gaussian_noise(x, self.gauss_noise)
            
        if self.rand_seg_remove > 0:
            
            rem = np.random.choice(np.unique(seg), self.rand_seg_remove, replace=False)
            for i in rem:
                x[seg == i] = 0

        return x, y
