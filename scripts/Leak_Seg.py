
# -*- coding: utf-8 -*-

from DataLoaders.Seg_DataLoader import Seg_DataLoader
from Generators.Seg_Generator import Seg_Generator
from Models.UNet3D import UNet3D_Extra
from Metrics.metrics import weighted_dice_coefficient_loss
from pred_AAA import pred_AAA
from config import config
import nibabel as nib
from DataUtils.Seg_tools import fast_process
import Metrics.eval_metrics as metrics
from Callbacks.LR_decay import get_callbacks
import numpy as np
from sklearn.model_selection import train_test_split
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
                 distort = False,
                 permute = True
                 )

   test_gen = Seg_Generator(data_points = test,
                 dim=(1, 128, 128, 128),
                 batch_size = 1,
                 n_classes = 1,
                 shuffle = False,
                 augment = False)
   
   return gen, test_gen

def get_pre_items(names):

    pre_names = [name.replace('art', 'pre') for name in names]
    pre_names = [pre_name.replace('ven', 'pre') for pre_name in pre_names]

    dl_pre = Seg_DataLoader(init_location='/media/sage/data/nifti_endoleak/',
                            label_location=main_dr + 'labels/leak_segs/',
                            annotations=main_dr + 'labels/annotations.csv',
                            label_type='crop',
                            seg_key='garbage',
                            n_classes=1,
                            neg_list=pre_names,
                            in_memory=False,
                            memory_loc='/mnt/sda5/temp/',
                            preloaded=False)

    items = dl_pre.get_all()

    return items

print('Running')

TRAIN = False
EVAL = True
SAVE = True

add_extra = False

folds = 5
epochs = 200
threshold = .5

main_dr = '/home/sage/GenDiagFramework/'
loss_func = weighted_dice_coefficient_loss

dl = Seg_DataLoader(
        init_location = '/media/sage/data/nifti_endoleak/',
        label_location =  main_dr + 'labels/leak_segs/',
        annotations = main_dr + 'labels/annotations.csv',
        neg_list = None,
        in_memory = False,
        memory_loc = '/mnt/sda5/temp/',
        preloaded=False)

print('Loading training data+labels')
dl.setup_kfold_splits(folds, 43)

if TRAIN:
    for fold in range(1, folds):
        
        train, test = dl.get_k_split(fold)

        tr, val = train_test_split(train, test_size=.15, random_state=43)
        gen, test_gen = create_gens(tr, val)
    
        model = UNet3D_Extra(input_shape = (1, 128, 128, 128), n_labels=1)
        model.compile(optimizer=keras.optimizers.adam(.001), loss=loss_func)
        
        callbacks = get_callbacks(model_file = main_dr + 'saved_models/Leak-' + str(fold) + '.h5',
                                  initial_learning_rate=5e-3,
                                  learning_rate_drop=.5,
                                  learning_rate_epochs=None,
                                  learning_rate_patience=10,
                                  verbosity=1,
                                  early_stopping_patience=30)
            
        model.fit_generator(
                        generator=gen,
                        validation_data=test_gen,
                        use_multiprocessing=True,
                        workers=8,
                        epochs=epochs,
                        callbacks = callbacks)
        
        #model.save_weights(main_dr + 'saved_models/Leak-' + str(fold) + '.h5')
        
if EVAL:
    
    names = []

    pre_leak_results = []
    leak_results = []
    post_leak_results = []
    post_pre_leak_results = []

    vol_dict = {}
    label_dict = {}
    
    for fold in range(folds):
        
        model = UNet3D_Extra(input_shape = (1, 128, 128, 128), n_labels=1)
        model.compile(optimizer=keras.optimizers.adam(.001), loss=loss_func)
        
        train, test = dl.get_k_split(fold)

        extra = get_pre_items([t.get_name() for t in test])
        test += extra
        
        #if add_extra:
        #    train_neg, test_neg = dl_neg.get_k_split(fold)
        #    train = train + train_neg
        #    test = test + test_neg
        
        gen, test_gen = create_gens(train, test)
        model.load_weights(main_dr + 'saved_models/Leak-' + str(fold) + '.h5')
        
        print('*Predicting Leaks for fold - ', fold, '*')
        preds = model.predict_generator(test_gen)
        
        del model
        
        for p in range(len(preds)):
            
            #Load info for the test
            truth = np.squeeze(test[p].get_label(copy=True))
            name = test[p].get_name()
            pixdims = test[p].get_pixdims()

            label = False
            if np.sum(truth) > 1:
                label = True

            label_dict[name] = label
            
            print('---------')
            print(name, label)
            
            #Load the prediction
            pred = preds[p]
            
            pred[pred > threshold] = 1
            pred[pred < threshold] = 0
            
            pred = np.squeeze(pred)

            if label:
                pre_leak_results.append(compute_metrics(pred, truth))
                print('pre_leak_results: ', pre_leak_results[-1])

            pred = fast_process(pred, .1)
            
            if label:
                leak_results.append(compute_metrics(pred, truth))
                print('leak_results: ', leak_results[-1])
            
            print('*Predicting AAA*')
            AAA_dp = pred_AAA(name)
            AAA = AAA_dp.get_pred_label()
            
            AAA_intersect = np.sum(pred * AAA[0])
            pred_AAA = pred * AAA[0]

            if label:
                post_leak_results.append(compute_metrics(pred_AAA, truth))
                print('post_leak_results: ', post_leak_results[-1])

            AAA_intersect_volume = AAA_intersect * \
                (pixdims[0] * pixdims[1] * pixdims[2]) * 0.001 # .001 For mL
            print('AAA_intersect_volume: ', AAA_intersect_volume)

            True_volume = truth * (pixdims[0] * pixdims[1] * pixdims[2]) * 0.001
            print('True_volume: ', True_volume)
            
            vol_dict[name] = AAA_intersect_volume
            
            print('---------')
            print()
            
    for name in vol_dict:
        if 'pre' not in name:
            
            pre_name = name.replace('art', 'pre')
            pre_name = pre_name.replace('ven', 'pre')
            
            print(name, ': ', end='')
            print(vol_dict[name] - vol_dict[name], label_dict[name])
            
        
        
    
    '''
    print('Predicting AAA for all scans')
    AAA_preds = []
    for name in names:
        a_pr = pred_AAA([name])
        
        AAA_preds += a_pr
    
    #Use to index
    AAA_names = [a.get_name() for a in AAA_preds]
    Leak_names = [a.get_name() for a in all_dps]
    
    leak_results = []
    
    #Now, should have all AAA_preds in AAA_preds and all leak preds in all_dps-
    for dp in all_dps:
        name = dp.get_name()
        
        if 'pre' not in name:
            print('---------')
            print(name, ":")
            
            pixdims = dp.get_pixdims()
            
            truth = np.squeeze(dp.get_label(copy=True))
            pred = np.squeeze(dp.get_pred_label())
            
            label = 0
            
            #Only calculate metrics for ones w/ leak
            if np.sum(truth) > 1:
                leak_results.append(compute_metrics(pred, truth))
                print('leak_results - ', leak_results[-1])
                
                label = 1
            
            AAA_i = AAA_names.index(name)
            AAA = AAA_preds[AAA_i].get_pred_label()
            
            AAA_intersect = np.sum(pred * AAA[0])
            print('AAA intersect - ', AAA_intersect)
            
            AAA_intersect_volume = AAA_intersect * (pixdims[0] * pixdims[1] * pixdims[2]) * 0.001
            print('AAA intersect volume - ', AAA_intersect_volume)
            
            #Find + process pre_contrast scan
            print('Find pre_contrast pred')
            pre_name = name.replace('art', 'pre')
            pre_name = pre_name.replace('ven', 'pre')
            
            pre_i = Leak_names.index(pre_name)
            pre = all_dps[pre_i].get_pred_label(copy=True)
            pre_pixdims = all_dps[pre_i].get_pixdims()
            
            print('Find pre_constrast seg')
            pre_AAA_i = AAA_names.index(pre_name)
            pre_AAA = AAA_preds[pre_AAA_i].get_pred_label()
            
            pre_AAA_intersect = np.sum(pre * pre_AAA[0])
            print('Pre-AAA intersect - ', pre_AAA_intersect)
            
            pre_AAA_intersect_volume = pre_AAA_intersect * (pre_pixdims[0] * pre_pixdims[1] * pre_pixdims[2]) * 0.001
            print('Pre-AAA intersect volume - ', pre_AAA_intersect_volume)
            
            difference = AAA_intersect_volume - pre_AAA_intersect_volume
            print('Volume difference - ', difference)
            print('With label = ', label)
            
            print('---------')
            print()
            
            #if SAVE:
            #    affine = test[p].get_affine()
            
            #    final = nib.Nifti1Image(pred, affine)
            #    final.to_filename(main_dr + 'predictions/' + name + '_endo_pred.nii.gz')
            '''
                

    print('pre Leak Means = ', np.mean(pre_leak_results, axis=0))
    print('pre Leak stds = ', np.std(pre_leak_results, axis=0))
    print('Leak Means = ', np.mean(leak_results, axis=0))
    print('Leak stds = ', np.std(leak_results, axis=0))
    print('post Leak Means = ', np.mean(post_leak_results, axis=0))
    print('post Leak stds = ', np.std(post_leak_results, axis=0))
        
        
        
        


