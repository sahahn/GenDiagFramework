# -*- coding: utf-8 -*-

from Generators.Seg_Generator import Seg_Generator
import DataUtils.augment_3d as augment_3d
import numpy as np

class IQ_Generator(Seg_Generator):
            
    def data_generation(self, list_IDs_temp):

        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, self.n_classes))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            X[i], y[i] = self.get_sample_from(ID)
       
        return X, y
    
    def get_sample_from(self, ID):
        
        #If augment return a copy
        x = self.data_points[ID].get_data(copy=self.augment)
        y = self.data_points[ID].get_label(copy=self.augment)
        
        if self.augment:
            x,y = self.data_aug(x, y, self.data_points[ID].get_affine(copy=True))
            
        return x, y
        
    def data_aug(self, x, y, affine):
        
        if self.distort:
            x,y = augment_3d.augment_data(x, y, affine,
                      flip=self.flip, scale_deviation=self.dist_scale)
        
        if self.permute:
            
            #Assumes channel first
            if x.shape[-3] != x.shape[-2] or x.shape[-2] != x.shape[-1]:
                raise ValueError("To utilize permutations, data array must be in 3D cube shape with all the same length.")
                
            x, y = augment_3d.random_permutation_x_y(x, y)

        return x, y
