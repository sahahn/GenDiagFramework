# -*- coding: utf-8 -*-
from Generators.TwoD_Generator import TwoD_Generator
from RetinaNet.utils.anchors import anchors_for_shape, anchor_targets_bbox
import DataUtils.transform as transform

import numpy as np

class RN_Generator(TwoD_Generator):
        
    def data_aug(self, x, y):
        
        trans = transform.adjust_transform_for_image(
                next(self.transform_generator),
                x, self.transform_parameters.relative_translation)
        
        x = transform.apply_transform(trans, x, self.transform_parameters)
        y[:4] = transform.transform_aabb(trans, y[:4])
        
        x = np.expand_dims(x, axis=-1)
        return x, y
    
    def extra(self, X, y):
        
        anchors = anchors_for_shape(self.dim)
        targets = list(anchor_targets_bbox(anchors, X, y, self.n_classes))
        return X, targets
    
    
