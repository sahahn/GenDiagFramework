# -*- coding: utf-8 -*-
from DataLoaders.Par_Seg_DataLoader import Par_Seg_DataLoader
from Generators.Seg_Generator import Seg_Generator

from Models.UNet3D import UNet3D_Extra
from Metrics.metrics import par_weighted_dice_coefficient_loss
import Callbacks.snapshot as snap


import nibabel as nib

from DataUtils.Seg_tools import proc_prediction

import keras
import numpy as np


#Prediction

model = UNet3D_Extra(input_shape = (1, 128, 128, 128), n_labels=2)
model.compile(optimizer=keras.optimizers.adam(), loss=loss_func)

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
        
        test[i] = proc_prediction(test[i], preds[i])
        
        name = test[i].get_name()
        affine = test[i].get_affine()
        
        pred = test[i].get_pred_label()
        output = np.zeros(np.shape(pred[0]))
        
        for i in range(len(pred)):
            output[pred[i] == 1] = i+1
    
        final = nib.Nifti1Image(output, affine)
        final.to_filename(main_dr + 'predictions/' + name + '_pred.nii.gz')
        
        print(name)