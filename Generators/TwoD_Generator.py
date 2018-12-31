# -*- coding: utf-8 -*-


from Generators.Generator import Generator
import DataUtils.transform as transform

class TwoD_Generator(Generator):
    
    def __init__(self,
                 data_points,
                 dim,
                 batch_size,
                 n_classes,
                 shuffle,
                 augment,
                 label_size=1,
                 transform_generator=None,
                 transform_parameters=None):
        
        super().__init__(data_points, dim, batch_size, n_classes,
             shuffle, augment, label_size)
        
        if self.augment and transform_generator == None:
            transform_generator = transform.random_transform_generator(
                min_rotation=-0.15,
                max_rotation=0.15,
                min_translation=(-0.1, -0.1),
                max_translation=(0.1, 0.1),
                min_shear=-0.1,
                max_shear=0.1,
                min_scaling=(0.8, 0.8),
                max_scaling=(1.2, 1.2),
                flip_x_chance=0,
                flip_y_chance=0)
        
        if self.augment and transform_parameters == None:
            transform_parameters = transform.TransformParameters()
            
        self.transform_generator = transform_generator
        self.transform_parameters = transform_parameters

    
        