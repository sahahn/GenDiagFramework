# -*- coding: utf-8 -*-

from DataLoaders.Seg_DataLoader import Seg_DataLoader
from Generators.Seg_Generator import Seg_Generator
from Models.UNet3D import UNet3D_Extra
from Metrics.metrics import dice_coefficient_loss
from config import config
import Callbacks.snapshot as snap

import keras


TRAIN = False
EVAL = True
SAVE = False

main_dr = '/home/sage/GenDiagFramework/'
loss_func = dice_coefficient_loss

dl = Seg_DataLoader(
        init_location = '/media/sage/data/nifti_endoleak/',
        label_location =  main_dr + 'labels/leak_segs/',
        annotations = main_dr + 'labels/annotations.csv',
        in_memory = True,
        memory_loc = config['memory_loc'])

def create_gens(train, test):
    
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
                 shuffle = False,
                 augment = False)
   
   return gen, test_gen



folds = 5
epochs = 100
num_snaps = 5

dl.setup_kfold_splits(folds, 43)

if TRAIN:
    for fold in range(2, folds):
        
        train, test = dl.get_k_split(fold)
        gen, test_gen = create_gens(train, test)
    
        model = UNet3D_Extra(input_shape = (1, 128, 128, 128), n_labels=1)
        model.compile(optimizer=keras.optimizers.adam(), loss=loss_func)
    
        snapshot = snap.SnapshotCallbackBuilder(epochs, num_snaps, .01)
        model_prefix = main_dr + 'saved_models/Leak_Fold-%d' % (fold)
            
        model.fit_generator(
                        generator=gen,
                        use_multiprocessing=True,
                        workers=8,
                        epochs=epochs,
                        callbacks=snapshot.get_callbacks(model_prefix=model_prefix)
                        )


