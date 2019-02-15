
# -*- coding: utf-8 -*-

import keras
import numpy as np
from DataLoaders.Flat_DataLoader import Flat_DataLoader
from Generators.BC_Generator import BC_Generator
import keras.applications
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from Models.Resnet50s import ResNet50 as RN_smaller
from Callbacks.LR_decay import get_callbacks
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from keras.models import Model
from keras.layers import Dense

import DataUtils.transform as transform
import os

#os.system('export HDF5_USE_FILE_LOCKING=FALSE')

load = False
map_inds = [0, 1, 2, 3, 4, 5]
#[thickness, area, avg_curv, curv, volume, sulc]
#   0          1       2       3     4       5

input_dims = (300, 600, 2*len(map_inds))
TRAIN = True

initial_lr = .0001
num_to_load = None
epochs = 60

main_dr = '/home/sage/GenDiagFramework/'
model_loc = main_dr + 'saved_models/flat_iq.h5'
temp_loc = '/mnt/sdb2/temp/'

preloaded = False
bs = 4
scale_labels = False

if TRAIN:
    file_dr = '/mnt/sda5/flat_maps/'
else:
    file_dr = '/mnt/sda5/flat_maps/'

def create_gens(train, test):

    gen, test_gen = None, None

    if train != None:   

        tg = transform.random_transform_generator(
                    min_rotation=-0.075,
                    max_rotation=0.075,
                    min_translation=(-0.001, -0.001),
                    max_translation=(0.001, 0.001),
                    min_shear=-0.001,
                    max_shear=0.001,
                    min_scaling=(0.99, 0.99),
                    max_scaling=(1.01, 1.01),
                    flip_x_chance=0,
                    flip_y_chance=0)


        gen = BC_Generator(
                 data_points = train,
                 dim = input_dims,
                 batch_size = bs,
                 n_classes = 1,
                 shuffle = True,
                 augment = False,
                 label_size = 1,
                 transform_generator = tg,
                 transform_parameters = None)

                
                

    if test != None:
        test_gen = BC_Generator(
                 data_points = test,
                 dim = input_dims,
                 batch_size = 1,
                 n_classes = 1,
                 shuffle = False,
                 augment = False,
                 label_size = 1)

    if gen != None and test_gen != None:
        return gen, test_gen
    
    elif gen != None:
        return gen
    
    elif test_gen != None:
        return test_gen


dl = Flat_DataLoader(
                 init_location = file_dr,
                 label_location = main_dr + 'labels/ABCD_labels.csv',
                 input_size = map_inds, #Weird yes, but lazy I am
                 limit = num_to_load,
                 scale_labels = scale_labels,
                 in_memory = False,
                 memory_loc = temp_loc,
                 compress = False,
                 preloaded = preloaded
                 )

base_model = Xception(include_top=False, weights=None, input_shape = input_dims, classes=1, pooling = 'avg')
#base_model = ResNet50(include_top=False, weights=None, input_shape = input_dims, classes=1, pooling = 'max')
#base_model = InceptionV3(include_top=False, weights=None, input_shape = input_dims, classes=1, pooling = 'avg')

x = base_model.output
output_layer = Dense(1)(x)
model = Model(inputs=base_model.input, outputs=output_layer)

if TRAIN:
    train, test = dl.get_train_test_split(.2, 43)
    print(len(train), len(test))

    model.compile(loss = 'mean_squared_error', optimizer = keras.optimizers.adam(initial_lr))

    if load:
        model.load_weights(model_loc)
        print('loaded weights')

    model.summary()
    gen, test_gen = create_gens(train, test)

    callbacks = get_callbacks(model_file = model_loc,
                            initial_learning_rate=initial_lr,
                            learning_rate_drop=.5,
                            learning_rate_epochs=None,
                            learning_rate_patience=50,
                            verbosity=1,
                            early_stopping_patience=130)

    model.fit_generator(generator=gen,
                        validation_data=test_gen,
                        use_multiprocessing=True,
                        workers=8,
                        epochs=epochs,
                        callbacks=callbacks)

else:
    test = dl.get_all()
    print(len(test))

    model.load_weights(model_loc)
    print('loaded weights')

    test_gen = create_gens(None, test)

    preds = model.predict_generator(test_gen, workers=8, verbose=1)
    for p in range(len(preds)):
        test[p].set_pred_label(float(preds[p]))
    
    true = [dp.get_label() for dp in test]
    pred = [dp.get_pred_label() for dp in test]
    
    r2_score = r2_score(true, pred)
    print('r2 score: ', r2_score)

    mae_score = mean_absolute_error(true, pred)
    print('mae score: ', mae_score)

    mse_score = mean_squared_error(true, pred)
    print('mse score: ', mse_score)
                            
                            
