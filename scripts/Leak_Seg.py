# -*- coding: utf-8 -*-

from DataLoaders.Seg_DataLoader import Seg_DataLoader
from Generators.Seg_Generator import Seg_Generator
from Models.UNet3D import UNet3D_Extra
from Metrics.metrics import weighted_dice_coefficient_loss
from config import config
import nibabel as nib
import Metrics.eval_metrics as metrics
import numpy as np

import keras

def compute_metrics(pred, truth):
     dc = metrics.dice_coef(pred, truth)
     iou = metrics.IOU(pred, truth)
     
     return [dc, iou]

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


TRAIN = True
EVAL = False
SAVE = True

main_dr = '/home/sage/GenDiagFramework/'
loss_func = weighted_dice_coefficient_loss

dl = Seg_DataLoader(
        init_location = '/media/sage/data/nifti_endoleak/',
        label_location =  main_dr + 'labels/leak_segs/',
        annotations = main_dr + 'labels/annotations.csv',
        neg_list = main_dr + 'labels/neg_leak_list.txt',
        in_memory = False,
        memory_loc = config['memory_loc'],
        preloaded=False)


folds = 5
epochs = 30
threshold = .5

dl.setup_kfold_splits(folds, 43)

if TRAIN:
    for fold in range(0, folds):
        
        train, test = dl.get_k_split(fold)

        for t in train:
            print(np.shape(t.get_label()), t.get_name())

        gen, test_gen = create_gens(train, test)
    
        model = UNet3D_Extra(input_shape = (1, 128, 128, 128), n_labels=1)
        model.compile(optimizer=keras.optimizers.adam(.001), loss=loss_func)
            
        model.fit_generator(
                        generator=gen,
                        use_multiprocessing=False,
                        workers=1,
                        epochs=epochs
                        )
        
        model.save_weights(main_dr + 'saved_models/Leak-' + str(fold) + '.h5')
        
if EVAL:
    
    model = UNet3D_Extra(input_shape = (1, 128, 128, 128), n_labels=1)
    model.compile(optimizer=keras.optimizers.adam(.001), loss=loss_func)
    
    results = []
    
    for fold in range(folds):
        
        train, test = dl.get_k_split(fold)
        
        gen, test_gen = create_gens(train, test)
        model.load_weights(main_dr + 'saved_models/Leak-' + str(fold) + '.h5')
        
        preds = model.predict_generator(test_gen)
        
        
        for p in range(len(preds)):
            
            name = test[p].get_name()
            pixdims = test[p].get_pixdims()
            
            truth = np.squeeze(test[p].get_label(copy=True))
            pred = np.squeeze(preds[p])
            
            pred[pred > threshold] = 1
            pred[pred < threshold] = 0
            results.append(compute_metrics(pred, truth))
            
            print(name)
            #print(metrics.volume(pred, pixdims), metrics.volume(truth, pixdims))
            print(results[-1])
            print('----')
            
            if SAVE:
                affine = test[p].get_affine()
            
                final = nib.Nifti1Image(pred, affine)
                final.to_filename(main_dr + 'predictions/' + name + '_endo_pred.nii.gz')
                
                
    print('Leak Means = ', np.mean(results, axis=0))
    print('Leak stds = ', np.std(results, axis=0))
        
        
        
        
        


