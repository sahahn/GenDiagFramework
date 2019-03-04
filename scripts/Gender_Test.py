
# -*- coding: utf-8 -*-

import keras
import numpy as np
from DataLoaders.IQ_DataLoader import IQ_DataLoader
from Generators.IQ_Generator import IQ_Generator
from Models.Resnet3D import Resnet3DBuilder
from Models.CNNs_3D import CNN_3D
from Callbacks.LR_decay import get_callbacks
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
import os

os.system('export HDF5_USE_FILE_LOCKING=FALSE')

input_dims = (160, 192, 160, 1)
TRAIN = False

initial_lr = .0001
num_to_load = None
epochs = 60

load_saved_weights = False

main_dr = '/home/sage/GenDiagFramework/'
model_loc = main_dr + 'saved_models/Gender_2.h5'
temp_loc = '/home/sage/temp/'

preloaded = True
bs = 4
scale_labels = False

if TRAIN:
    file_dr = '/home/sage/training/'
else:
    file_dr = '/home/sage/testing/'

def create_gens(train, test):

    gen, test_gen = None, None

 
    if train != None:   
        gen = IQ_Generator(data_points = train,
                dim=input_dims,
                batch_size = bs,
                n_classes = 1,
                shuffle = True,
                augment = True,
                distort = True,
                dist_scale = .05,
                flip = False,
                permute = False,
                gauss_noise = .001
                )

    if test != None:
        test_gen = IQ_Generator(data_points = test,
                dim=input_dims,
                batch_size = 1,
                n_classes = 1,
                shuffle = False,
                augment = False)

    if gen != None and test_gen != None:
        return gen, test_gen
    
    elif gen != None:
        return gen
    
    elif test_gen != None:
        return test_gen



dl = IQ_DataLoader(
                 init_location = file_dr,
                 label_location = main_dr + 'labels/ABCD_genders.csv',
                 input_size = input_dims,
                 load_segs = True,
                 limit = num_to_load,
                 iq = True,
                 scale_labels = scale_labels,
                 in_memory = False,
                 memory_loc = temp_loc,
                 compress = False,
                 preloaded = preloaded
                 )

rn_builder = Resnet3DBuilder()
model = rn_builder.build_resnet_18(input_shape=input_dims, num_outputs=1, reg_factor=1e-4, regression=False)
#model = CNN_3D(input_dims, 4, 0, False)

if TRAIN:
    train, test = dl.get_train_test_split(.2, 43)
    print(len(train), len(test))

    model.compile(loss = 'binary_crossentropy', optimizer = keras.optimizers.adam(initial_lr), metrics=['accuracy'])

    if load_saved_weights:

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
    
    print('roc auc: ', roc_auc_score(true, pred))
    
    pred = np.array(pred).round()
    print('f1 score: ', f1_score(true, pred))
    print('precision score: ', precision_score(true, pred))
    print('recall_score: ',  recall_score(true, pred))
    print('acc : ', accuracy_score(true, pred))
                            
                            
from DataLoaders.IQ_DataLoader import IQ_DataLoader 


main_dr = '/home/sage/GenDiagFramework/' 
dl = IQ_DataLoader( 
                     init_location = file_dr, 
                     label_location = main_dr + 'labels/ABCD_genders.csv', 
                     input_size = (160, 192, 160, 1), 
                     load_segs = True, 
                     limit = 10, 
                     in_memory = True, 
                     memory_loc = None, 
                     compress = False, 
                     preloaded = False 
                     )                        