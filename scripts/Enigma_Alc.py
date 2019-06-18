import keras
import numpy as np
import os, sys

from DataLoaders.ABCD_DataLoader import ABCD_DataLoader
from Generators.IQ_Generator import IQ_Generator
from Models.Resnet3D import Resnet3DBuilder
from Models.CNNs_3D import CNN_3D
from Callbacks.LR_decay import get_callbacks
from Callbacks.ROC_callback import ROC_callback
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score

np.warnings.filterwarnings('ignore')
os.system('export HDF5_USE_FILE_LOCKING=FALSE')

input_dims = (192, 192, 192, 1)
main_dr    = '/home/sage/GenDiagFramework/'
model_loc  = main_dr + 'saved_models/Alc.h5'

TRAIN              = True
load_saved_weights = False
bs                 = 4
initial_lr         = .0001

def create_train_gen(train):
    
    gen = IQ_Generator(
                    data_points     = train,
                    dim             = input_dims,
                    batch_size      = bs,
                    n_classes       = 1,
                    shuffle         = False,
                    augment         = False,
                    distort         = True,
                    dist_scale      = .05,
                    gauss_noise     = .001,
                    rand_seg_remove = 0
                    )

    return gen

def create_test_gen(test):

    test_gen = IQ_Generator(
                    data_points     = test,
                    dim             = input_dims,
                    batch_size      = 1,
                    n_classes       = 1,
                    shuffle         = False,
                    augment         = False
                    )

    return test_gen

dl = ABCD_DataLoader(
                    init_location   = '/home/sage/enigma',
                    label_location  = main_dr + 'labels/Alc_Subjects.csv',
                    label_key       = '',
                    load_extra_info = True,
                    file_key        = 'brain.finalsurfs.mgz',
                    input_size      = input_dims,
                    load_segs       = False,
                    segs_key        = 'aparc.a2009s+aseg.mgz',
                    min_max         = (0,255),
                    limit           = None,
                    in_memory       = False,
                    memory_loc      = '/home/sage/alc-temp/',
                    compress        = False,
                    preloaded       = True
                    )

model = CNN_3D(input_dims,
                    sf              = 4,
                    n_layers        = 6,
                    d_rate          = 0, 
                    batch_norm      = True, 
                    regression      = False, 
                    coord_conv      = False
                    )

model.compile(
                    loss ='binary_crossentropy',
                    optimizer=keras.optimizers.adam(initial_lr),
                    metrics=['accuracy', auc_roc, auc_1]
                    )

model.summary()
train, test, val = dl.get_splits_by_site(test_sites=[5], val_sites=[3])
print('Train size: ', len(train))
print('Test  size: ', len(test))
print('Val   size: ', len(val))

if TRAIN:

    if load_saved_weights:
        model.load_weights(model_loc)
        print('loaded weights')

    train_gen, test_gen = create_train_gen(train), create_test_gen(val)

    callbacks = get_callbacks(
                    model_file              = model_loc,
                    initial_learning_rate   = initial_lr,
                    learning_rate_drop      = .5,
                    learning_rate_epochs    = None,
                    learning_rate_patience  = 50,
                    verbosity               = 1,
                    early_stopping_patience = 130
                    )

    callbacks.append(ROC_callback(train_gen, val_gen, train, val, workers=8))

    model.fit_generator(
                    generator               = train_gen,
                    validation_data         = test_gen,
                    use_multiprocessing     = True,
                    workers                 = 8,
                    epochs                  = 50,
                    callbacks               = callbacks
                    )

else:
  
    model.load_weights(model_loc)
    test_gen = create_test_gen(test)

    pred = model.predict_generator(test_gen, workers=8, verbose=1)
    true = np.array([dp.get_label() for dp in test])
    print('roc auc: ', roc_auc_score(true, pred))
    
    pred = np.array(pred).round()
    print('f1 score: ', f1_score(true, pred))
    print('precision score: ', precision_score(true, pred))
    print('recall_score: ',  recall_score(true, pred))
    print('acc : ', accuracy_score(true, pred))
