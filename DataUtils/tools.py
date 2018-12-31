# -*- coding: utf-8 -*-

import numpy as np
from skimage.transform import resize
import scipy
import scipy.ndimage

def normalize_data(scan):

    scan = scan.astype('float32')
    scan -= np.mean(scan)
    scan /= np.max(scan)

    imax = np.max(scan)
    imin = np.min(scan)

    scan -= imin
    scan /= (imax-imin)
    
    return scan


def resample(image, new_shape):

    
    resize_factor = np.array(new_shape) / image.shape
    image = scipy.ndimage.interpolation.zoom(image, resize_factor)
    
    return image


def resample_3d(image, affine, new_shape):
    
    scale_factor = np.array(new_shape) / image.shape
    image = scipy.ndimage.interpolation.zoom(image, scale_factor)
    
    new_affine = np.copy(affine)
    new_affine[:3, :3] = affine[:3, :3] * scale_factor
    new_affine[:, 3][:3] = affine[:, 3][:3] + (image.shape * np.diag(affine)[:3] * (1 - scale_factor)) / 2
    
    return image, new_affine

def determine_crop(label, image, new_shape, pad_limit):
    ''' Returns a mix of crop/resized version of an image
    
    label - should be in [x1, y1, x2, y2] representing a crop on the image,
    image - underlying image to return a crop  
    new_shape - desired final crop shape/ returned image shape
    pad_limit - the maximum the image should be padded in any dimension before reshape

    '''
    
    shp = np.shape(image)
    label_n = label.copy()
    
    range1 = label_n[3] - label_n[1] + 1
    range2 = label_n[2] - label_n[0] + 1
    
    counter1 = 0
    counter2 = 0
    
    flip = True
    
    while range1 < new_shape[0] and counter1 < pad_limit:
        if flip and label_n[3] < shp[0]:
            label_n[3] += 1
            flip = False
            counter1+=1
        elif not flip and label_n[1] > 0:
            label_n[1] -= 1
            flip = True
            counter1+=1
            
    while range2 < new_shape[0] and counter2 < pad_limit:
        if flip and label_n[2] < shp[1]:
            label_n[2] += 1
            flip = False
            counter2+=1
        elif not flip and label_n[0] > 0:
            label_n[0] -= 1
            flip = True
            counter2+=1
            
    slc = image[label_n[1]:label_n[3]+1, label_n[0]:label_n[2]+1]
    
    if np.shape(slc) != new_shape:
        slc = resize(slc, new_shape, mode='constant')
        
    return slc


def update(slc, l, rng):
    '''Cleverness maybe, why are you reading this?'''
    
    rng = np.array(rng)
    
    rng[[0,2,3]] = np.min([rng[[0,2,3]], [slc, l[0], l[1]]], axis=0)
    rng[[1,4,5]] = np.max([rng[[1,4,5]], [slc, l[2], l[3]]], axis=0)
    
    return [int(r) for r in rng]

def calculate_ranges(data_points):
    '''Given input as a list of data_points, w/ RN style labels-
       label = [x0, y0, x1, y2, class]
       create a dictionary that maps scan name to
       in terms of axial [bot, top, x0, y0, x1, y1]
    '''
    
    range_dict = {}
    
    for dp in data_points:
        
        name = dp.name
        slc = dp.slc
        label = dp.get_label()
        
        if name in range_dict:
            range_dict[name] = update(slc, label[:-1], range_dict[name])
        else:
            range_dict[name] = [int(slc), int(slc), int(label[0]),
                      int(label[1]), int(label[2]), int(label[3])]
            
            
    return range_dict

def determine_crop_3d(scan, ranges, thickness, new_shape, base_pad, axial_pad, affine):
    ''' 
    scan - the scan (3d matrix) to crop/reshape
    ranges - in terms of axial [bot, top, x0, y0, x1, y1]
    thickness - the slice thickness of the scan
    new_shape - the desired new shape (In format (Sag, Cor. Axial))
    base_pad - num to pad in sag. + cor. dimension
    axial_pad - num to pad in axial dimension (pre divided by thickness
    affine - the affine of the scan)
    '''
    
    a_p = int(axial_pad / thickness)
    
    d1 = (ranges[4] + base_pad) - (ranges[2] - base_pad)
    d2 = (ranges[5] + base_pad) - (ranges[3] - base_pad)
    d3 = (ranges[1] + a_p) - (ranges[0] - a_p)

    flip = True

    while d1 < new_shape[0]:
        if flip:
            ranges[4] += 1
            flip = False
        else:
            ranges[2] -= 1
            flip = True
        d1 += 1

    while d2 < new_shape[1]:
        if flip:
            ranges[5] += 1
            flip = False
        else:
            ranges[3] -= 1
            flip = True
        d2 += 1

    while d3 < new_shape[2]:
        if flip:
            ranges[1] += 1
            flip = False
        else:
            ranges[0] -= 1
            flip = True
        d3 += 1

    scan = scan[ranges[2] - base_pad : ranges[4] + base_pad, 
                ranges[3] - base_pad : ranges[5] + base_pad,
                ranges[0] - a_p : ranges[1] + a_p ]
    
    shp = np.shape(scan)

    if shp != new_shape:
        scan, affine = resample_3d(scan, affine, new_shape)

    return scan, affine
    
    
    
    
        
        
                


