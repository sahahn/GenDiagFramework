# -*- coding: utf-8 -*-
from DataLoaders.Par_Seg_DataLoader import Par_Seg_DataLoader
from Generators.Seg_Generator import Seg_Generator

from Models.UNet3D import UNet3D_Extra
from Metrics.metrics import par_weighted_dice_coefficient_loss
import Callbacks.snapshot as snap

import keras

main_dr = '/home/sage/GenDiagFramework/'
loss_func = par_weighted_dice_coefficient_loss

def create_gens(train, test):
    
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
    
    return gen, test_gen

dl = Par_Seg_DataLoader(
        init_location = '/media/sage/data/nifti_endoleak/',
        label_location = '/home/sage/EndoLeak/Segmentation/nifti-crop/',
        annotations = main_dr + 'labels/annotations.csv',
        label_type='crop',
        seg_key=['seg', 'full'],
        n_classes=2,
        in_memory = True)

folds = 5
epochs = 100

dl.setup_kfold_splits(folds, 43)


#Training
for fold in range(2, folds):
    train, test = dl.get_k_split(fold)
    
    gen, test_gen = create_gens(train, test)

    model = UNet3D_Extra(input_shape = (1, 128, 128, 128), n_labels=2)
    model.compile(optimizer=keras.optimizers.SGD(), loss=loss_func)

    snapshot = snap.SnapshotCallbackBuilder(epochs, 5, .01)
    model_prefix = main_dr + 'saved_models/AAA_Fold-%d' % (fold)
        
    model.fit_generator(
                    generator=gen,
                    use_multiprocessing=True,
                    workers=8,
                    epochs=epochs,
                    callbacks=snapshot.get_callbacks(model_prefix=model_prefix)
                    )
    
    
    
    
    


