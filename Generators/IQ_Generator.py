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
        
    def data_aug(self, x, y, affine):
        
        if self.distort:
            x = augment_3d.augment_just_data(x, affine,
                      flip=self.flip, scale_deviation=self.dist_scale)
            
        if self.permute:
            x = augment_3d.random_permutation_x(x)
            
        if self.gauss_noise != 0:
            x = augment_3d.add_gaussian_noise(x, self.guass_noise)
            
        return x, y
