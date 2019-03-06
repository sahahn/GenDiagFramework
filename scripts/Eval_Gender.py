# -*- coding: utf-8 -*-

import keras
import numpy as np
from DataLoaders.IQ_DataLoader import IQ_DataLoader
from Models.CNNs_3D import CNN_3D

input_dims = (160, 192, 160, 1)

main_dr = '/home/sage/GenDiagFramework/'
model_loc = main_dr + 'saved_models/Gender.h5'
temp_loc = '/home/sage/temp/'
file_dr = '/home/sage/testing/'

preloaded = True

dl = IQ_DataLoader(
                 init_location = file_dr,
                 label_location = main_dr + 'labels/ABCD_genders.csv',
                 input_size = input_dims,
                 load_segs = True,
                 in_memory = False,
                 memory_loc = temp_loc,
                 compress = False,
                 preloaded = preloaded
                 )


model = CNN_3D(input_dims, 4, 0, False)
model.load_weights(model_loc)

test = dl.get_all()

 #x[seg == i] = 0

for i in range(len(test)):
    
    dp = test[i]
    
    seg = dp.get_guide_label()
    label = dp.get_label()
    
    baseline = model.predict(np.expand_dims(dp.get_data(), axis=0))[0][0]
    
    print(i)
    
    with open('results.csv', 'a') as f:
        
        f.write(str(baseline))
        f.write(',')
        f.write(str(label))
        f.write(',')
    
        for j in range(1,109):
            
            data = dp.get_data()
            data[seg==j] = 0
            
            score = model.predict(np.expand_dims(data, axis=0))[0][0]
            f.write(str(score))
            f.write(',')
            
        f.write('\n')
        
    
    

