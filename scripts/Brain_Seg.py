# -*- coding: utf-8 -*-

#clips = [[-50, 200], [-50, 300], [-100, 400], [-800, np.max(x)/3], [-100, 200]]



# -*- coding: utf-8 -*-

from DataLoaders.BrainCT_DataLoader import BrainCT_DataLoader
from Generators.Seg_Generator import Seg_Generator
from Models.UNet3D import UNet3D_Extra
from Metrics.metrics import weighted_dice_coefficient_loss
import Metrics.eval_metrics as metrics
from Callbacks.LR_decay import get_callbacks
import numpy as np
import Metrics.eval_metrics as metrics
from sklearn.model_selection import train_test_split
import keras

def compute_metrics(pred, truth):
     dc = metrics.dice_coef(pred, truth)
     iou = metrics.IOU(pred, truth)
     
     return [dc, iou]

def create_gens(train, test):
    
   gen = Seg_Generator(data_points = train,
                 dim=(5, 32, 256, 256),
                 batch_size = 1,
                 n_classes = 1,
                 shuffle = True,
                 augment = True,
                 distort = True,
                 permute = False)

   test_gen = Seg_Generator(data_points = test,
                 dim=(5, 32, 256, 256),
                 batch_size = 1,
                 n_classes = 1,
                 shuffle = False,
                 augment = False)
   
   return gen, test_gen

print('Running')

TRAIN = False
EVAL = True
SAVE = False

add_extra = False

folds = 5
epochs = 200
threshold = .5

main_dr = '/home/sage/GenDiagFramework/'
loss_func = weighted_dice_coefficient_loss

input_size = (5, 32, 256, 256)

dl = BrainCT_DataLoader(
        init_location =  main_dr + 'labels/brain_nifti/',
        label_location =  main_dr + 'labels/brain_nifti/',
        annotations = None,
        label_type='crop',
        seg_key='seg',
        n_classes=1,
        neg_list = None,
        seg_input_size=input_size,
        clip_range = [[-50, 200], [-50, 300], [-100, 400], [-800, 2200], [-100, 200]],
        in_memory = True)

print('Loading training data+labels')
dl.setup_kfold_splits(folds, 43)

if TRAIN:
    for fold in range(0, folds):
        
        train, test = dl.get_k_split(fold)

        tr, val = train_test_split(train, test_size=.15, random_state=43)
        gen, test_gen = create_gens(tr, val)
    
        model = UNet3D_Extra(input_shape = input_size, n_labels=1)
        model.compile(optimizer=keras.optimizers.adam(.001), loss=loss_func)
        
        callbacks = get_callbacks(model_file = main_dr + 'saved_models/Brain-' + str(fold) + '.h5',
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
        
if EVAL:
    
    dcs = []
    
    model = UNet3D_Extra(input_shape = input_size, n_labels=1)
    model.compile(optimizer=keras.optimizers.adam(.001), loss=loss_func)
    
    for fold in range(0, folds):
        
        train, test = dl.get_k_split(fold)
        gen, test_gen = create_gens(train, test)
        
        model.load_weights(main_dr + 'saved_models/Brain-' + str(fold) + '.h5')
        
        preds = model.predict_generator(test_gen)
        
        for p in range(len(preds)):
            pred = preds[p]
            truth = test[p].get_label(True)
            
            dc = metrics.dice_coef(pred, truth)
            dcs.append(dc)
            
            print(dc)
        
    print(np.mean(dcs, axis=0))
    print(np.std(dcs, axis=0))
        
        
    
        
    