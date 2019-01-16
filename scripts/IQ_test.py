# -*- coding: utf-8 -*-


import keras
import numpy as np
from DataLoaders.IQ_DataLoader import IQ_DataLoader
from Generators.IQ_Generator import IQ_Generator
from Models.Resnet3D import Resnet3DBuilder


input_dims = (256, 256, 256, 1)

def create_gens(train, test):
    
   gen = IQ_Generator(data_points = train,
                 dim=input_dims,
                 batch_size = 1,
                 n_classes = 1,
                 shuffle = True,
                 augment = False,
                 distort = False,
                 dist_scale = .1,
                 permute = False
                 )

   test_gen = IQ_Generator(data_points = test,
                 dim=input_dims,
                 batch_size = 1,
                 n_classes = 1,
                 shuffle = False,
                 augment = False)
   
   return gen, test_gen


dl = IQ_DataLoader(
                 init_location = '/media/sage/Images/training/',    
                 label_location = '/home/sage/Neuro/ABCD_Challenge/training_fluid_intelligenceV1.csv',
                 seg_input_size = input_dims,
                 limit=50,
                 in_memory=True,
                 memory_loc=None,
                 compress=False,
                 preloaded=False
                 )

train, test = dl.get_train_test_split(.2, 43)

print(len(train), len(test))

rn_builder = Resnet3DBuilder()
model = rn_builder.build_resnet_34(input_shape=input_dims, num_outputs=1, reg_factor=1e-4)
model.compile(loss = 'mean_squared_error', optimizer = keras.optimizers.adam(.001))

gen, test_gen = create_gens(train, test)

model.fit_generator(generator=gen,
                    validation_data=test_gen,
                    use_multiprocessing=True,
                    workers=8,
                    epochs=20)
                            
                            
