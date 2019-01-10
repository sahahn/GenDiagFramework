
# -*- coding: utf-8 -*-

from DataLoaders.Seg_DataLoader import Seg_DataLoader
from Generators.Seg_Generator import Seg_Generator
from Models.UNet3D import UNet3D_Extra
from Metrics.metrics import weighted_dice_coefficient_loss
from config import config
import nibabel as nib
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


TRAIN = True
EVAL = False
SAVE = True

main_dr = '/home/sage/GenDiagFramework/'
loss_func = weighted_dice_coefficient_loss

dl = Seg_DataLoader(
        init_location = '/media/sage/data/nifti_endoleak/',
        label_location =  main_dr + 'labels/leak_segs/',
        annotations = main_dr + 'labels/annotations.csv',
        neg_list = None,
        in_memory = True,
        memory_loc = config['memory_loc'],
        preloaded=False)


folds = 5
epochs = 200
threshold = .5

dl.setup_kfold_splits(folds, 43)

if TRAIN:
    for fold in range(1, folds):
        
        train, test = dl.get_k_split(fold)

        tr, val = train_test_split(train, test_size=.15, random_state=43)

        #for t in train:
        #    print(np.shape(t.get_label()), t.get_name())

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
        
        
        
        
        


