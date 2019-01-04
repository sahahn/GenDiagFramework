# -*- coding: utf-8 -*-
from DataLoaders.Par_Seg_DataLoader import Par_Seg_DataLoader
from Generators.Seg_Generator import Seg_Generator

from Models.UNet3D import UNet3D_Extra
from Metrics.metrics import par_weighted_dice_coefficient_loss
import Callbacks.snapshot as snap

import Metrics.eval_metrics as ev_metrics
import nibabel as nib

from DataUtils.Seg_tools import process

import keras
import numpy as np

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

folds = 2
epochs = 100
num_snaps = 5

dl.setup_kfold_splits(folds, 43)

#Training
for fold in range(2, folds):
    
    train, test = dl.get_k_split(fold)
    gen, test_gen = create_gens(train, test)

    model = UNet3D_Extra(input_shape = (1, 128, 128, 128), n_labels=2)
    model.compile(optimizer=keras.optimizers.adam(), loss=loss_func)

    snapshot = snap.SnapshotCallbackBuilder(epochs, num_snaps, .01)
    model_prefix = main_dr + 'saved_models/AAA_Fold-%d' % (fold)
        
    model.fit_generator(
                    generator=gen,
                    use_multiprocessing=True,
                    workers=8,
                    epochs=epochs,
                    callbacks=snapshot.get_callbacks(model_prefix=model_prefix)
                    )


#Prediction

model = UNet3D_Extra(input_shape = (1, 128, 128, 128), n_labels=2)
model.compile(optimizer=keras.optimizers.adam(), loss=loss_func)

threshold = .5

for fold in range(folds):
    
    train, test = dl.get_k_split(fold)
    gen, test_gen = create_gens(train, test)
    
    preds = []
    
    for s in range(1, num_snaps+1):
        
        model.load_weights(main_dr + 'saved_models/AAA_Fold-' + str(fold)
                           + '-' + str(s) + '.h5')
        
        preds.append(model.predict_generator(test_gen))
    
    preds = np.mean(preds, axis=0)
    
    for i in range(len(test)):
        
        name = test[i].get_name()
        affine = test[i].get_affine()
        
        pred = preds[i]
        
        x = np.squeeze(pred[0,])
        y = np.squeeze(pred[1,])
        
        output = np.zeros((128,128,128))
        
        x[x < threshold] = 0
        y[y < threshold] = 0
    
        output[x>y] = 1
        output[y>x] = 2
        
        output = process(output)
    
        final = nib.Nifti1Image(output, affine)
        final.to_filename(main_dr + 'predictions/' + name + '_pred.nii.gz')
        print(name)

        
        

    
    
    
    
    
    


