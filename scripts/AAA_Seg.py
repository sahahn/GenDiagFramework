# -*- coding: utf-8 -*-
from DataLoaders.Par_Seg_DataLoader import Par_Seg_DataLoader
from Generators.Seg_Generator import Seg_Generator

from Models.UNet3D import UNet3D_Extra
from Metrics.metrics import par_weighted_dice_coefficient_loss
import Metrics.eval_metrics as metrics
import Callbacks.snapshot as snap
from DataUtils.max_ap import calculate_max_axial_ap

import nibabel as nib

from DataUtils.Seg_tools import proc_prediction

import keras
import numpy as np

main_dr = '/home/sage/GenDiagFramework/'
loss_func = par_weighted_dice_coefficient_loss

TRAIN = False
EVAL = True
SAVE = False

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

def compute_metrics(pred, truth, pixdims):
     dc = metrics.dice_coef(pred, truth)
     iou = metrics.IOU(pred, truth)
     abs_dif, percent_dif = metrics.volume_dif(pred, truth, pixdims)
     
     return [dc, iou, abs_dif, percent_dif]
                    
     

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
num_snaps = 5

dl.setup_kfold_splits(folds, 43)

#Training

if TRAIN:
    for fold in range(0, folds):
        
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

if EVAL:
    model = UNet3D_Extra(input_shape = (1, 128, 128, 128), n_labels=2)
    model.compile(optimizer=keras.optimizers.adam(), loss=loss_func)
    
    #3 dif. preds
    AAA_results = []
    Graft_results = []
    Both_results = []
    
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
            
            truth = test[i].get_label(copy=True)
            
            #Make sure it is a full scan!
            if np.sum(truth[-1]) > (128 * 128 * 128) - 2000:
                
                #Sets test pred label to proc. version
                test[i] = proc_prediction(test[i], preds[i])
                name = test[i].get_name()
                pred = test[i].get_pred_label(copy=True)
                pixdims = test[i].get_pixdims()
                
                AAA_results.append(compute_metrics(pred[0], truth[0], pixdims))
                Graft_results.append(compute_metrics(pred[1], truth[1], pixdims))
                
                both_pred = np.zeros(np.shape(pred[0]))
                both_truth = np.zeros(np.shape(pred[0]))
                
                both_pred[pred[0]+pred[1] > 0] = 1
                both_truth[truth[0]+truth[1] > 0] = 1
                
                Both_results.append(compute_metrics(both_pred, both_truth, pixdims))
                
                #print(name, AAA_results[-1], Graft_results[-1], Both_results[-1])
                
                print(pixdims)
                pred_max_ap = calculate_max_axial_ap(both_pred, pixdims)
                truth_max_ap = calculate_max_axial_ap(both_truth, pixdims)
                
                print(name)
                print(pred_max_ap)
                print(truth_max_ap)
                print()
                
    
                if SAVE:
                
                    pred = test[i].get_pred_label(copy=True)
                    name = test[i].get_name()
                    affine = test[i].get_affine()
            
                    output = np.zeros(np.shape(pred[0]))
                    
                    for i in range(len(pred)):
                        output[pred[i] == 1] = i+1
                
                    final = nib.Nifti1Image(output, affine)
                    final.to_filename(main_dr + 'predictions/' + name + '_pred.nii.gz')
                    
                    print('saved ', name)
    
    print('AAA Means = ', np.mean(AAA_results, axis=0))
    print('AAA stds = ', np.std(AAA_results, axis=0))
    
    print('Graft Means = ', np.mean(Graft_results, axis=0))
    print('Graft stds = ', np.std(Graft_results, axis=0))
    
    print('Both Means = ', np.mean(Both_results, axis=0))
    print('Both stds = ', np.std(Both_results, axis=0))

    
    
    
    
    
    


