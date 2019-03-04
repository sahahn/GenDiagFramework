# -*- coding: utf-8 -*-
import numpy as np

def get_crop_ind(image, rtol= 1e-08):

    coords = np.argwhere(image > rtol)

    xs = coords.min(axis=0)
    ys = coords.max(axis=0) + 1

    return xs, ys


def fill_to(data, input_size):

    shp = np.shape(data)

    if (shp[:-1] < np.array(input_size)[:-1]).all():
        new_data = np.zeros(input_size)
        
        shp = np.array(shp)
        shp2 = np.array(input_size)
        
        d = np.floor((shp2 - shp)/2).astype(int)
        new_data[d[0]:shp[0]+d[0], d[1]:shp[1]+d[1], d[2]:shp[2]+d[2]] = data
        
        return new_data

    else:
        return data