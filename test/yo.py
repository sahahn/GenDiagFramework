# -*- coding: utf-8 -*-

from DataLoaders.Seg_DataLoader import Seg_DataLoader
from Generators.Seg_Generator import Seg_Generator
from Models.UNet3D import UNet3D_Extra
from Metrics.metrics import dice_coef_loss
from config import config

import keras

dl = Seg_DataLoader(
        init_location = '/media/sage/data/nifti_endoleak/',
        label_location = '/home/sage/GenDiagFramework/labels/leak_segs/',
        annotations = '/home/sage/GenDiagFramework/labels/annotations.csv')

train, test = dl.get_train_test_split(.2, 43)


gen = Seg_Generator(data_points = train,
                 dim=(1, 128, 128, 128),
                 batch_size = 1,
                 n_classes = 1,
                 shuffle = True,
                 augment = True,
                 distort = True,
                 )

test_gen = Seg_Generator(data_points = test,
                 dim=(1, 128, 128, 128),
                 batch_size = 1,
                 n_classes = 1,
                 shuffle = True,
                 augment = False)


model = UNet3D_Extra(input_shape = (1, 128, 128, 128), n_labels=1)
model.compile(optimizer=keras.optimizers.adam(lr=.001), loss=dice_coef_loss)

callbacks =  [keras.callbacks.ModelCheckpoint(config['model_loc'] + '/model-{epoch:02d}.h5')]

model.fit_generator(generator=gen, validation_data=test_gen,
                            use_multiprocessing=True,
                            workers=8,
                            epochs=1, 
                            callbacks=callbacks)
