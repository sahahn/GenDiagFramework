import keras
import numpy as np

from DataLoaders.ABCD_DataLoader import ABCD_DataLoader
from Generators.IQ_Generator import IQ_Generator
from Models.Resnet3D import Resnet3DBuilder
from Models.CNNs_3D import CNN_3D
from Callbacks.LR_decay import get_callbacks
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os, sys

np.warnings.filterwarnings('ignore')
os.system('export HDF5_USE_FILE_LOCKING=FALSE')

TRAIN = True

initial_lr = .0001
num_to_load = None
epochs = 60

input_dims = (192, 192, 192, 1)
main_dr = '/home/sage/GenDiagFramework/'
load_saved_weights = True

model_loc = main_dr + 'saved_models/Gender.h5'
temp_loc = '/home/sage/temp/'

preloaded = True
bs = 1

def create_gens(train, test):

    gen, test_gen = None, None

    if train != None:   
        gen = IQ_Generator(
                data_points = train,
                dim=input_dims,
                batch_size = bs,
                n_classes = 1,
                shuffle = False,
                augment = True,
                distort = True,
                dist_scale = .05,
                gauss_noise = .001,
                rand_seg_remove = 0
                )

    if test != None:
        test_gen = IQ_Generator(
                data_points = test,
                dim=input_dims,
                batch_size = 1,
                n_classes = 1,
                shuffle = False,
                augment = False
                )

    if gen != None and test_gen != None:
        return gen, test_gen
    
    elif gen != None:
        return gen
    
    elif test_gen != None:
        return test_gen

dl = ABCD_DataLoader(
                 init_location = '/home/sage/FS2/FS_data_release_2/',
                 label_location = main_dr + 'labels/Gender_Train.csv',
                 label_key='NDAR',
                 file_key='brain.finalsurfs.mgz',
                 input_size=input_dims,
                 load_segs=False,
                 segs_key='aparc.a2009s+aseg.mgz',
                 tal_transform=False,
                 tal_key='talairach.xfm',
                 limit=num_to_load,
                 in_memory=False,
                 memory_loc=temp_loc,
                 compress=True,
                 preloaded=preloaded
                 )

rn_builder = Resnet3DBuilder()
#model = rn_builder.build_resnet_50(input_shape=input_dims, num_outputs=1, reg_factor=1e-4, regression=False)
#model = CNN_3D(input_dims, 6, .1, False)
#model = CNN_3D(input_dims, 8, .2, False)
#model = rn_builder.build_resnet_18(input_shape=input_dims, num_outputs=1, reg_factor=1e-4, regression=False)


model = CNN_3D(input_dims, 4, .2, False) #Gender
#model = CNN_3D(input_dims, 6, .3, False) #Gender2
#model = CNN_3D(input_dims, 8, .25, False) #Gender3
#model = CNN_3D(input_dims, 10, .2, False) #Gender4
#model = CNN_3D(input_dims, 2, .2, False) #Gender5

#test = dl.get_all()

train, test, val = dl.get_train_test_val_split(.2, .1, 43)
#print(len(train), len(test), len(val))

model.compile(loss = 'binary_crossentropy', optimizer = keras.optimizers.adam(initial_lr), metrics=['accuracy'])

if TRAIN:

    if load_saved_weights:

        model.load_weights(model_loc)
        print('loaded weights')

    model.summary()
    gen, test_gen = create_gens(train, val)

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
  
    all_preds = []
   
#    for m,n in zip(['', '2', '4'], [4,6,10]):
    for m,n in zip(['', '2', '3', '4', '5'], [4, 6, 8, 10, 2]):

        model_loc = main_dr + 'saved_models/Gender' + m + '.h5'
        model = CNN_3D(input_dims, n, .2, False)

        model.load_weights(model_loc)
        test_gen = create_gens(None, test)
        preds = model.predict_generator(test_gen, workers=8, verbose=1)

        for p in range(len(preds)):
            test[p].set_pred_label(float(preds[p]))

        true = np.array([dp.get_label() for dp in test])
        pred = np.array([dp.get_pred_label() for dp in test])
        all_preds.append(pred)


    '''
    model.load_weights(model_loc)

    test_gen = create_gens(None, test)
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
    '''
    '''

    model_loc = main_dr + 'saved_models/Gender6.h5'
    model = CNN_3D(input_dims, 4, .2, False)

    test_gen = create_gens(test, None)

    all_test_preds = []
    for x in range(20):

        preds = model.predict_generator(test_gen, workers=8, verbose=1)

        for p in range(len(preds)):
            test[p].set_pred_label(float(preds[p]))

        true = np.array([dp.get_label() for dp in test])
        pred = np.array([dp.get_pred_label() for dp in test])
        all_test_preds.append(pred)
    
    all_preds.append(np.mean(all_test_preds, axis=0))
    '''
    stds = np.std(all_preds, axis=0)
    print(np.mean(stds))
    pred = np.mean(all_preds, axis=0)
  
    #print(stds)
    #print(true)
    #print(pred)
    print('roc auc: ', roc_auc_score(true, pred))
    
    pred = np.array(pred).round()

    print('f1 score: ', f1_score(true, pred))
    print('precision score: ', precision_score(true, pred))
    print('recall_score: ',  recall_score(true, pred))
    print('acc : ', accuracy_score(true, pred))

    
   
