# -*- coding: utf-8 -*-

import numpy as np

def dice_coef(pred, label):
    '''Given a prediction and ground truth label, return the dice coef.'''
    
    smooth = 0.00001
    return (2 * np.sum(pred * label) + smooth) / (np.sum(label) + np.sum(pred))

def IOU(pred, label):
    '''Given a prediction and ground truth label, return the intersection over
       union score.'''
       
    return np.sum(pred * label) / ((np.sum(label) + np.sum(pred)) - np.sum(pred * label))

def volume(data, pixdims):
    ''' Calculates the volume in mL of a segmented CT scan
    
        data - The segmented volume, w/ segmentation class == 1
        pixdims - The measurments for pixdims and slice thickness as an array
     
    '''
    
    cubic_mm = len(data[data==1]) * (pixdims[0] * pixdims[1] * pixdims[2])
    mL = cubic_mm * 0.001
    
    return mL

def volume_dif(pred, label, pixdims):
    
    pred_volume = volume(pred, pixdims)
    label_volume = volume(label, pixdims)
    
    dif = abs(label_volume - pred_volume)
    percent_dif = dif / label_volume
    
    return dif, percent_dif
                        