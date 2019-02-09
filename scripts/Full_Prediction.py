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
from DataUtils.max_ap import calculate_max_axial_ap

import nibabel as nib
import copy
import os, random

np.warnings.filterwarnings('ignore')

main_dr = '/home/sage/GenDiagFramework/'
data_loc = '/home/sage/nifti_endoleak/'

RN_input_size = (512, 512, 1)
Seg_input_size = (1, 128, 128, 128)

RN_thresh = .75

folds = 5
num_snaps = 5

num_to_get = 1
SAVE = True

bs = 6

already_pred = os.listdir(main_dr + 'predictions/')
al_p = [file.replace('_pred.nii.gz', ' ') for file in already_pred]

files = os.listdir(data_loc)
files = [file for file in files if '_art.nii' in file or '_ven.nii' in file]
files = [file.replace('.nii', '').replace('.gz', '') for file in files]
files = [file for file in files if file not in al_p]

all_names = random.choices(files, k = num_to_get)
all_dps = []

for name in all_names:
    print('start - ', name)

    names = [name]

    RN_dl = RN_DataLoader(
            init_location = data_loc,
            label_location = names,
            in_memory = False,
            memory_loc = main_dr + '/labels/temp/'
            )

    RN_dps = RN_dl.load_unseen()

    rem = len(RN_dps) % bs
    if rem > 0:
        RN_dps1 = RN_dps[:-rem]
    else:
        RN_dps1 = RN_dps

    gen1 = RN_Generator(data_points = RN_dps1,
                dim=RN_input_size,
                batch_size = bs,
                n_classes = 1,
                shuffle = False,
                augment = False,
                label_size = 5)

    model = load_RN_model(main_dr +'saved_models/RN.h5')
    boxes, scores = get_predictions(model, gen1, RN_thresh)

    if rem > 0:

        RN_dps2 = RN_dps[-rem:]

        gen2 = RN_Generator(data_points = RN_dps2,
                    dim=RN_input_size,
                    batch_size = 1,
                    n_classes = 1,
                    shuffle = False,
                    augment = False,
                    label_size = 5)
        
        boxes2, scores2 = get_predictions(model, gen2, RN_thresh)
        boxes += boxes2
        scores += scores2

    for i in range(len(boxes)):
        RN_dps[i].set_pred_label(boxes[i])

    RN_pred_dps = []

    destroy_thr = .2
    sec_thr = .5
    sec_num = 3

    while len(RN_pred_dps) == 0:

        print(sec_thr, end='')

        RN_dps_copy = copy.deepcopy(RN_dps)
        RN_dps_copy = post_process_boxes(RN_dps_copy, destroy_thr, sec_thr, sec_num)

        for dp in RN_dps_copy:
        
            try:
                label = list(dp.get_pred_label()[0]) + [1]
                dp.set_label(label)

                RN_pred_dps.append(dp)
            except:
                pass

        destroy_thr -= .01
        sec_thr -= .01

        if destroy_thr % .1 == 0:
            sec_num+=1

        if sec_thr < 0:
            print('WARNING!', names)
            print(scores)
            break

    del RN_dps
    #del model
    
    all_dps += RN_pred_dps
    print(names)

try:
    del model
except:
    pass

print('Loading Segs')
    
from Models.UNet3D import UNet3D_Extra

Seg_dl = Seg_DataLoader(
        init_location = data_loc,
        label_location =  main_dr + 'labels/leak_segs/',
        annotations = all_dps,
        seg_key = 'Garbage',
        n_classes = 2,
        neg_list = all_names,
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

print('Starting AAA Predictions')
    
for fold in range(folds):
    for s in range(1, num_snaps+1):
            
        model.load_weights(main_dr + 'saved_models/AAA_Fold-' + str(fold)
                           + '-' + str(s) + '.h5')
        preds.append(model.predict_generator(Seg_gen))
        
preds = np.mean(preds, axis=0)

for i in range(len(Seg_dps)):
    Seg_dps[i] = proc_prediction(Seg_dps[i], preds[i])
    
    pred = Seg_dps[i].get_pred_label()
    pixdims = Seg_dps[i].get_pixdims()
    name = Seg_dps[i].get_name()
    
    both_pred = np.zeros(np.shape(pred[0]), dtype=int)
    both_pred[pred[0]+pred[1] > 0] = 1
    
    pred_max_ap = calculate_max_axial_ap(both_pred, pixdims)
    print(name, pred_max_ap)
    
    with open('pred_aps', 'a') as f:
        
        f.write(name)
        f.write(' ')
        f.write(str(pred_max_ap[0]))
        f.write('-')
        f.write(str(pred_max_ap[1]))
        f.write('\n')
    
    if SAVE:
                
        pred = Seg_dps[i].get_pred_label(copy=True)
        affine = Seg_dps[i].get_affine()

        output = np.zeros(np.shape(pred[0]))
        
        for i in range(len(pred)):
            output[pred[i] == 1] = i+1
    
        final = nib.Nifti1Image(output, affine)
        final.to_filename(main_dr + 'predictions/' + name + '_pred.nii.gz')
        
        print('saved ', name)


                

    
    
