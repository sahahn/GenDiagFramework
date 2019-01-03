# -*- coding: utf-8 -*-

import numpy as np
from skimage import measure
from skimage import segmentation

def get_s_e(seg, start_flag):

    #Arbitrary high start number, used as a flag
    start_num = 10000
    end_num = 10000

    for l in range(len(seg)):
        
        cnt = 0
        for i in range(1, start_flag):
            cnt += np.count_nonzero(seg[l] == i)
    
        if start_num == 10000 and cnt != 0:
            start_num = l

        if start_num != 10000 and end_num == 10000 and cnt == 0:
            end_num = l

    return start_num, end_num

def get_seen(seg, start_flag):
    '''
       seg - 3D Segmentation input
       start_flag - The number (seg color) associated with the axial completed
                    slices. S.t. Sagittal flag = start_flag+1, and
                    Coronal = start_flag+2.
                    Likewise it is assumed that 1 to start_flag-1 represent valid 
                    classes to which are 'seen'
       
       Provided a 3D partial segmentation, calculate the a 3D seen array.
       Input is assumed to begin as Sagittal first.
    '''
    
    seen = np.zeros(np.shape(seg))
    
    #Visit each of the 3 dimensions
    for x in range(3):

        flag = start_flag+1 #Assumed sag. start input

        #For axial
        if x == 1: 
            flag = start_flag

            seg = seg.transpose(2,0,1)
            seen = seen.transpose(2,0,1)
        
        #For coronal
        elif x == 2:
            flag = start_flag+2

            seg = seg.transpose(2,1,0)
            seen = seen.transpose(2,1,0)

        #Helper fuction to calculate seen range for a given dimension
        s,e = get_s_e(seg, start_flag)
        
        #Based on the calculated start (s) and end (e) for that dimension, set
        #seen accordinglu
        seen[:s] = 1
        
        #Arbitraryly high number used in s_e to represent no value found
        if e != 10000:
            seen[e:] = 1

        for l in range(len(seg)):
            if flag in seg[l]:
                seen[l] = 1
    
    seg = seg.transpose(1,0,2)
    seen = seen.transpose(1,0,2)
    
    
    return seen

def fast_process(p):
    ''' p - Predicted 3D segmentation, with 1 or more classes.
    
        Return a post processed version, attempting to remove outliers. This
        is notably the faster and rougher version for post proccessing segs.'''
    
    labels = measure.label(p)
    highest = np.max(np.unique(labels))

    total = len(labels[labels > -1]) - len(labels[labels==0])
    
    for i in range(1, highest+1):
        percent = len(labels[labels == i]) / total
        if percent < .05:
            p[labels == i] = 0
            
    return p

def custom_bound(i, labels):
    
    l = np.array(labels, copy=True) 
    l[l != i] = 0
    
    return segmentation.find_boundaries(l, connectivity=1, mode='outer', background=0)


def process(p):
    '''p - Predicted 3D segmentation, with 1 or more classes.
       Return a post processed version, attempting to remove outliers.
    '''
    
    p[p == 0] = 300
    
    labels = measure.label(p)
    highest = np.max(np.unique(labels))

    total = len(labels[labels > -1])

    for i in range(1, highest+1):
        
        num = len(labels[labels == i])
        percent = num / total
        
        #Default hard coded param for now...
        if percent < .01:
            p[labels == i] = np.argmax(np.bincount(p[custom_bound(i, labels)].astype(np.int)))

    p[p == 300] = 0
    
    return p



