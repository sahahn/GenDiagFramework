# -*- coding: utf-8 -*-

from Generators.TwoD_Generator import TwoD_Generator
import DataUtils.transform as transform

import numpy as np

class BC_Generator(TwoD_Generator):
        
    def data_aug(self, x, y):
        
        trans = transform.adjust_transform_for_image(
                next(self.transform_generator),
                x, self.transform_parameters.relative_translation)
        
        x = transform.apply_transform(trans, x, self.transform_parameters)        
        #x = np.expand_dims(x, axis=-1)
        
        return x, y
        