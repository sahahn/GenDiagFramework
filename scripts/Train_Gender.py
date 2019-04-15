import keras
import numpy as np

from DataLoaders.ABCD_DataLoader import ABCD_DataLoader
from Generators.IQ_Generator import IQ_Generator
from Models.Resnet3D import Resnet3DBuilder
from Models.CNNs_3D import CNN_3D
from Callbacks.LR_decay import get_callbacks
import os
np.warnings.filterwarnings('ignore')
os.system('export HDF5_USE_FILE_LOCKING=FALSE')

TRAIN = True

initial_lr = .0001
num_to_load = None
epochs = 60

input_dims = (192, 192, 192, 1)
main_dr = '/home/sage/GenDiagFramework/'
load_saved_weights = False

model_loc = main_dr + 'saved_models/Gender.h5'
temp_loc = '/home/sage/temp/'

preloaded = False
bs = 2

def create_gens(train, test):

    gen, test_gen = None, None

    if train != None:   
        gen = IQ_Generator(
                data_points = train,
                dim=input_dims,
                batch_size = bs,
                n_classes = 1,
                shuffle = True,
                augment = False,
                distort = True,
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
                 compress=False,
                 preloaded=preloaded
                 )

rn_builder = Resnet3DBuilder()
model = rn_builder.build_resnet_50(input_shape=input_dims, num_outputs=1, reg_factor=1e-4)
#model = CNN_3D(input_dims, 8, .1, False)


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
