# -*- coding: utf-8 -*-

from DataLoaders.IQ_DataLoader import IQ_DataLoader
import numpy as np


dl = IQ_DataLoader(
                 init_location = '/media/sage/Images/training/',    
                 label_location = '/home/sage/Neuro/ABCD_Challenge/training_fluid_intelligenceV1.csv',
                 limit=50,
                 in_memory=True,
                 memory_loc=None,
                 compress=False,
                 preloaded=False
                 )

train, test = dl.get_train_test_split(.2, 43)

for t in train:
    print(np.shape(t.get_data()))