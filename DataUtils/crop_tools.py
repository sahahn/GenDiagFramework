# -*- coding: utf-8 -*-
import numpy as np

def get_crop_ind(image, rtol= 1e-08):

    coords = np.argwhere(image > rtol)

    xs = coords.min(axis=0)
    ys = coords.max(axis=0) + 1

    return xs, ys


def fill_to(data, input_size):

    shp = np.shape(data)

    print(np.shape(data))

    if (shp[:-1] < np.array(input_size)[:-1]).all():
        new_data = np.zeros(input_size)

        print(np.shape(new_data))

        d1 = np.floor((np.shape(new_data) - np.shape(data))/2)
        d2 = np.ceil((np.shape(new_data) - np.shape(data))/2)
        
        new_data[d1[0]:shp[0]+d2[0], d1[1]:shp[1]+d2[1], d1[2]:shp[2]+d2[2]] = data
        
        return new_data

    else:
        return data