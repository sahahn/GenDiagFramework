# -*- coding: utf-8 -*-

from Generators.Generator import Generator
import numpy as np

class Seg_Generator(Generator):

    def data_generation(self, list_IDs_temp):

        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, self.n_classes, *self.dim[1:]))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            X[i], y[i] = self.get_sample_from(ID)
       
        return X, y
        

        