import keras
import numpy as np
import tensorflow as tf
import os, sys

import keras.backend as K

from DataLoaders.ABCD_DataLoader import ABCD_DataLoader
from Generators.IQ_Generator import IQ_Generator
from Models.Resnet3D import Resnet3DBuilder
from Models.CNNs_3D import CNN_3D
from Callbacks.LR_decay import get_callbacks
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score

def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.metrics.auc(y_true, y_pred)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value

def auc_1(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

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

    gen, test_gen = create_train_gen(train), create_test_gen(val)

    callbacks = get_callbacks(
                    model_file              = model_loc,
                    initial_learning_rate   = initial_lr,
                    learning_rate_drop      = .5,
                    learning_rate_epochs    = None,
                    learning_rate_patience  = 50,
                    verbosity               = 1,
                    early_stopping_patience = 130
                    )

    model.fit_generator(
                    generator               = gen,
                    validation_data         = test_gen,
                    use_multiprocessing     = True,
                    workers                 = 8,
                    epochs                  = 50,
                    callbacks               = callbacks
                    )

else:
  
    model.load_weights(model_loc)
    test_gen = create_test_gen(test)

    preds = model.predict_generator(test_gen, workers=8, verbose=1)
    
    for p in range(len(preds)):
        test[p].set_pred_label(float(preds[p]))

    true = np.array([dp.get_label() for dp in test])
    pred = np.array([dp.get_pred_label() for dp in test])
    print('roc auc: ', roc_auc_score(true, pred))
    
    pred = np.array(pred).round()
    print('f1 score: ', f1_score(true, pred))
    print('precision score: ', precision_score(true, pred))
    print('recall_score: ',  recall_score(true, pred))
    print('acc : ', accuracy_score(true, pred))
