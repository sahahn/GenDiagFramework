# -*- coding: utf-8 -*-

import DataLoaders.BC_DataLoader as BC_DataLoader
import Generators.BC_Generator as BC_Generator
from Models.Resnet50s import ResNet50

from config import config


data_loader = BC_DataLoader(
        init_location = '/media/sage/data/nifti_endoleak/',
        label_location = '/home/sage/GenDiagFramework/labels/AAArange.csv',
        in_memory = False,
        memory_loc = config['memory_loc'],
        compress = True)

train, test = data_loader.get_train_test_split(.2, 43)

print(len(train), len(test))

train_gen = BC_Generator(data_points = train,
                             dim=(128, 128, 1),
                             batch_size = 1,
                             n_classes = 1,
                             shuffle = True,
                             augment = True,
                             label_size = 5)

model = ResNet50()


