import keras
import numpy as np
import os, sys

from DataLoaders.ABCD_DataLoader import ABCD_DataLoader
from Generators.IQ_Generator import IQ_Generator
from Models.Resnet3D import Resnet3DBuilder
from Models.CNNs_3D import CNN_3D
from Callbacks.LR_decay import get_callbacks
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score

np.warnings.filterwarnings('ignore')
os.system('export HDF5_USE_FILE_LOCKING=FALSE')

TRAIN = True

initial_lr = .0001
num_to_load = None
epochs = 60

input_dims = (192, 192, 192, 1)
main_dr = '/home/sage/GenDiagFramework/'
load_saved_weights = False

model_loc = main_dr + 'saved_models/Alc.h5'
temp_loc = '/home/sage/alc-temp/'

preloaded = False
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
                augment = False,
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
                 init_location = '/home/sage/enigma',
                 label_location = main_dr + 'labels/Alc_Subjects.csv',
                 label_key='',
                 file_key='brain.finalsurfs.mgz',
                 input_size=input_dims,
                 load_segs=False,
                 segs_key='aparc.a2009s+aseg.mgz',
                 min_max=(0,255),
                 limit=num_to_load,
                 in_memory=False,
                 memory_loc=temp_loc,
                 compress=True,
                 preloaded=preloaded
                 )


#rn_builder = Resnet3DBuilder()
#model = rn_builder.build_resnet_50(input_shape=input_dims, num_outputs=1, reg_factor=1e-4, regression=False)
model = CNN_3D(input_dims, sf=4, n_layers=6, d_rate=0, batch_norm=True, regression=False, coord_conv=True)
model.summary()

#test = dl.get_all()
train, test, val = dl.get_train_test_val_split(.2, .1, 43)
print(len(train), len(test), len(val))

model.compile(loss ='binary_crossentropy', optimizer=keras.optimizers.adam(initial_lr), metrics=['accuracy'])

if TRAIN:

    if load_saved_weights:
        model.load_weights(model_loc)
        print('loaded weights')

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
