# -*- coding: utf-8 -*-
from DataLoaders.RN_DataLoader import RN_DataLoader
from Generators.RN_Generator import RN_Generator
from Models.RetinaNet import load_RN_model
from RetinaNet.retinanet import get_predictions
from DataUtils.RN_tools import post_process_boxes

import keras
import numpy as np
from Metrics.metrics import par_weighted_dice_coefficient_loss

from DataLoaders.Seg_DataLoader import Seg_DataLoader
from Generators.Seg_Generator import Seg_Generator
from DataUtils.Seg_tools import proc_prediction


main_dr = '/home/sage/GenDiagFramework/'
data_loc = '/mnt/sdb2/data/nifti_endoleak/'

RN_input_size = (512, 512, 1)
Seg_input_size = (1, 128, 128, 128)

RN_thresh = .75

folds = 5
num_snaps = 5

names = ['9a_art']


RN_dl = RN_DataLoader(
        init_location = data_loc,
        label_location = names,
        in_memory = True 
        )

RN_dps = RN_dl.load_unseen()

gen = RN_Generator(data_points = RN_dps,
             dim=RN_input_size,
             batch_size = 1,
             n_classes = 1,
             shuffle = False,
             augment = False,
             label_size = 5)

model = load_RN_model(main_dr +'saved_models/RN.h5')

boxes, scores = get_predictions(model, gen, RN_thresh)
del model

from Models.UNet3D import UNet3D_Extra

for i in range(len(boxes)):
    RN_dps[i].set_pred_label(boxes[i])
    
RN_dps = post_process_boxes(RN_dps)

RN_pred_dps = []

for dp in RN_dps:
    
    try:
        label = list(dp.get_pred_label()[0]) + [1]
        dp.set_label(label)
        
        RN_pred_dps.append(dp)
    except:
        pass

del RN_dps

Seg_dl = Seg_DataLoader(
        init_location = data_loc,
        label_location =  main_dr + 'labels/leak_segs/',
        annotations = RN_pred_dps,
        seg_key = 'Garbage',
        n_classes = 2,
        neg_list = names,
        in_memory = True)
    
Seg_dps = Seg_dl.get_all()

Seg_gen = Seg_Generator(
          data_points = Seg_dps,
          dim= Seg_input_size,
          batch_size = 1,
          n_classes = 2,
          shuffle = False,
          augment = False
          )

model = UNet3D_Extra(input_shape = Seg_input_size, n_labels=2)
model.compile(optimizer=keras.optimizers.adam(), loss=par_weighted_dice_coefficient_loss)

preds = []
    
for fold in range(folds):
    for s in range(1, num_snaps+1):
            
        model.load_weights(main_dr + 'saved_models/AAA_Fold-' + str(fold)
                           + '-' + str(s) + '.h5')
        preds.append(model.predict_generator(Seg_gen))
        
preds = np.mean(preds, axis=0)

for i in range(len(Seg_dps)):
    Seg_dps[i] = proc_prediction(Seg_dps[i], preds[i])

    
    
