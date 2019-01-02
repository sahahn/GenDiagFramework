# -*- coding: utf-8 -*-

from DataLoaders.Par_Seg_DataLoader import Par_Seg_DataLoader
from Generators.Seg_Generator import Seg_Generator

from Models.UNet3D import UNet3D_Extra
from Metrics.metrics import par_weighted_dice_coefficient_loss
from config import config

import keras

dl = Par_Seg_DataLoader(
        init_location = '/media/sage/data/nifti_endoleak/',
        label_location = '/home/sage/EndoLeak/Segmentation/nifti-crop/',
        annotations = '/home/sage/GenDiagFramework/labels/annotations.csv',
        label_type='crop',
        seg_key=['seg', 'full'],
        n_classes=2,
        in_memory = False,
        memory_loc = config['memory_loc'],
        compress = False)

train, test = dl.get_train_test_split(.2, 43)

gen = Seg_Generator(data_points = train,
                 dim=(1, 128, 128, 128),
                 batch_size = 1,
                 n_classes = 3,
                 shuffle = True,
                 augment = True,
                 distort = True,
                 )

test_gen = Seg_Generator(data_points = test,
                 dim=(1, 128, 128, 128),
                 batch_size = 1,
                 n_classes = 3,
                 shuffle = False,
                 augment = False)


model = UNet3D_Extra(input_shape = (1, 128, 128, 128), n_labels=2)
model.compile(optimizer=keras.optimizers.adam(lr=.001), loss=par_weighted_dice_coefficient_loss)

callbacks =  [keras.callbacks.ModelCheckpoint(config['model_loc'] + '/model-{epoch:02d}.h5')]

model.fit_generator(generator=gen, validation_data=test_gen,
                            use_multiprocessing=True,
                            workers=8,
                            epochs=1, 
                            callbacks=callbacks)
