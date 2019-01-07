# -*- coding: utf-8 -*-

import numpy as np
from scipy import ndimage
from numpy import nanmax, argmax, unravel_index
from scipy.spatial.distance import pdist, squareform
import math, scipy

def resample_to_spacing(image, spacing, new_spacing):
    
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    return image

def resample_seg(seg, pix_dims):
    '''Upsample segmentation, s.t. the pixel dimension are the same,
       and therefore the segmentation is easier to analyze is relation to
       distance.'''
    
    if pix_dims[0] != pix_dims[1]:
        new_dim = max(pix_dims[:2])
        seg = resample_to_spacing(seg, pix_dims, [new_dim, new_dim, pix_dims[2]])
    else:
        new_dim = pix_dims[0]
        
    return seg, new_dim


def get_max_dist(A):

    D = pdist(A)
    D = squareform(D);
    
    return nanmax(D), unravel_index(argmax(D), D.shape)

def get_y(a, b, c, x):
    
    return (c - (a * x)) / b

def calculateDistance(x1, y1, x2, y2):  
    
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2) 

def calc_perp_line(x1, x2, y1, y2):
    
    x3 = (x1+x2) // 2
    y3 = (y1+y2) // 2

    k = (((y2-y1) * (x3-x1)) - ((x2-x1) * (y3-y1))) / ((y2-y1)**2 + (x2-x1)**2)

    smll = 0.000001
    if k < smll or k > -smll:
        k = smll

    x4 = x3 - k * (y2-y1)
    y4 = y3 + k * (x2-x1)

    #Calculate the line forumla
    a = y3 - y4
    b = x4 - x3
    c = (a*x4) + (b*y4)
    
    return a,b,c

def get_approx_perp_points(points, a, b, c):
    '''Find the two points in the border closest to the actual perp line'''
    
    perps = []
    dif_val = 1

    while np.shape(perps) != (2,2):
        perps = []

        buffer = False
        buffer_cnt = 0

        for p in points:

            pred_y = int(round(get_y(a,b,c,p[1])))
            dif = abs(pred_y - p[0])
            
            if dif < dif_val and not buffer:
                perps.append(p)
                buffer = True

            if buffer:
                buffer_cnt += 1

            if buffer_cnt == 5:
                buffer = False

        dif_val += 1
        
    return perps[0][1], perps[1][1], perps[0][0], perps[1][0]

def get_max_ap(seg_slice, pix_dim, return_points=False):
    '''Helper function to calculate max ap on a given axial slice.'''
    
    #Calculate valid points on the edge of the segmentation
    dt = ndimage.distance_transform_edt(seg_slice)
    dt[dt > 1] = 0
    points = np.transpose((dt == 1).nonzero())
    
    #Slice must have atleast say 10 valid points
    if len(points) <= 10:
        return (0,0)
    
    #Calculate the points with the max_dist
    N, [I_row, I_col] = get_max_dist(points)
    
    xs = points[I_col][1], points[I_row][1]
    ys = points[I_col][0], points[I_row][0]

    y1, y2, x1, x2 = ys[0], ys[1], xs[0], xs[1]
    
    #Calculate the perpendicular line
    a,b,c = calc_perp_line(x1, x2, y1, y2)
    
    #Get approx perp. points
    px1, px2, py1, py2 = get_approx_perp_points(points, a, b, c)
    
    #Calculate the length of each line
    d1 = calculateDistance(x1,y1,x2,y2) * pix_dim
    d2 = calculateDistance(px1,py1,px2,py2) * pix_dim
    
    if return_points:
        return d1, d2, [x1, y1, x2, y2], [px1, py1, px2, py2]
    
    return d1,d2

def calculate_max_axial_ap(seg, dims, return_points=False):
    '''Given a seg as a 3D segmentation saved as [sag, cor, axial], return two
       measurements corresponding to Max AP. Optionally return the coordinates
       of the two lines'''

    seg[seg>0] = 1
    
    seg, dim = resample_seg(seg, dims)
    seg = seg.transpose(2,0,1)
    
    print(type(seg[0][0][0]))
    
    highest = 0
    ind = 0
    
    for i in range(len(seg)):
        
        try:
            d1,d2 = get_max_ap(seg[i], dim)
        except:
            d1,d2 = 0,0
        
        #In basic attempt to avoid edge cases... find highest of both lines
        if d1+d2 > highest and d1 < (d2*2):
            highest = d1+d2
            ind = i
    
    return get_max_ap(seg[ind], dim, return_points)
    
