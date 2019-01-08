# -*- coding: utf-8 -*-

from DataLoaders.Seg_DataLoader import Seg_DataLoader
from Generators.Seg_Generator import Seg_Generator
from Models.UNet3D import UNet3D_Extra
from Metrics.metrics import weighted_dice_coefficient_loss
from config import config
import nibabel as nib
import Metrics.eval_metrics as metrics

import keras

def compute_metrics(pred, truth, pixdims):
     dc = metrics.dice_coef(pred, truth)
     iou = metrics.IOU(pred, truth)
     abs_dif, percent_dif = metrics.volume_dif(pred, truth, pixdims)
     
     return [dc, iou, abs_dif, percent_dif]

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


TRAIN = False
EVAL = True
SAVE = True

main_dr = '/home/sage/GenDiagFramework/'
loss_func = weighted_dice_coefficient_loss

dl = Seg_DataLoader(
        init_location = '/media/sage/data/nifti_endoleak/',
        label_location =  main_dr + 'labels/leak_segs/',
        annotations = main_dr + 'labels/annotations.csv',
        in_memory = True,
        memory_loc = config['memory_loc'])


folds = 5
epochs = 30
threshold = .5

dl.setup_kfold_splits(folds, 43)

if TRAIN:
    for fold in range(0, folds):
        
        train, test = dl.get_k_split(fold)
        gen, test_gen = create_gens(train, test)
    
        model = UNet3D_Extra(input_shape = (1, 128, 128, 128), n_labels=1)
        model.compile(optimizer=keras.optimizers.adam(.001), loss=loss_func)
            
        model.fit_generator(
                        generator=gen,
                        use_multiprocessing=True,
                        workers=8,
                        epochs=epochs
                        )
        
        model.save_weights(main_dr + 'saved_models/Leak-' + str(fold) + '.h5')
        
if EVAL:
    
    dl_negs = Seg_DataLoader(
        init_location = '/media/sage/data/nifti_endoleak/',
        label_location =  main_dr + 'labels/leak_segs/',
        annotations = main_dr + 'labels/annotations.csv',
        label_type='crop', #Since just zeros dont resize
        seg_key='garbage', #Don't want it to get anything
        neg_list = main_dr + 'labels/neg_leak_list.txt',
        in_memory = False,
        memory_loc = config['memory_loc'])
    
    dl_negs.setup_kfold_splits(folds, 43)
    
    model = UNet3D_Extra(input_shape = (1, 128, 128, 128), n_labels=1)
    model.compile(optimizer=keras.optimizers.adam(.001), loss=loss_func)
    
    pre_results = []
    post_results = []

    for fold in range(folds):
        
        train, test = dl.get_k_split(fold)
        train_negs, test_negs = dl_negs.get_k_split(fold)
        
        train = train + train_negs
        test = test + test_negs
        
        gen, test_gen = create_gens(train, test)
        model.load_weights(main_dr + 'saved_models/Leak-' + str(fold) + '.h5')
        
        preds = model.predict_generator(test_gen)
        
        
        for p in range(len(preds)):
            
            name = test[p].get_name()
            pixdims = test[p].get_pixdims()
            truth = test[p].get_label(copy=True)
            pred = preds[p]
            
            print(name)
            pre_results.append(compute_metrics(pred, truth, pixdims))
            print(pre_results[-1])
            
            pred[pred > threshold] = 1
            post_results.append(compute_metrics(pred, truth, pixdims))
            print(post_results[-1])
            print()
            
            if SAVE:
                affine = test[p].get_affine()
            
                final = nib.Nifti1Image(pred, affine)
                final.to_filename(main_dr + 'predictions/' + name + '_endo_pred.nii.gz')
        
        
        
        
        


