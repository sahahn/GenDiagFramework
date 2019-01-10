# -*- coding: utf-8 -*-
'''Super shitty hard coded predictive code for now...'''



from DataLoaders.Seg_DataLoader import Seg_DataLoader
from Generators.Seg_Generator import Seg_Generator

from Models.UNet3D import UNet3D_Extra
from Metrics.metrics import par_weighted_dice_coefficient_loss
from DataUtils.Seg_tools import proc_prediction

import keras
import numpy as np


def pred_AAA(names):
    
    folds = 5
    num_snaps = 5
    
    main_dr = '/home/sage/GenDiagFramework/'
    
    dl = Seg_DataLoader(
        init_location = '/media/sage/data/nifti_endoleak/',
        label_location =  main_dr + 'labels/leak_segs/',
        annotations = main_dr + 'labels/annotations.csv',
        seg_key='Garbage',
        n_classes=2,
        neg_list = names,
        in_memory = False)
    
    to_eval = dl.get_all()
    
    eval_gen = Seg_Generator(data_points = to_eval,
                 dim=(1, 128, 128, 128),
                 batch_size = 1,
                 n_classes = 2,
                 shuffle = False,
                 augment = False)
    
    model = UNet3D_Extra(input_shape = (1, 128, 128, 128), n_labels=2)
    model.compile(optimizer=keras.optimizers.adam(), loss=par_weighted_dice_coefficient_loss)
    
    preds = []
    
    for fold in range(folds):
        for s in range(1, num_snaps+1):
                
                model.load_weights(main_dr + 'saved_models/AAA_Fold-' + str(fold)
                                   + '-' + str(s) + '.h5')
                
                preds.append(model.predict_generator(eval_gen))
            
    preds = np.mean(preds, axis=0)
    
    for i in range(len(to_eval)):
        to_eval[i] = proc_prediction(to_eval[i], preds[i])
        
    return to_eval
           

    
    
       
    
    
    
    