# -*- coding: utf-8 -*-

import keras
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, Decon

def encoder_model_200(input_shape=160,128,128):
    
    enc = Sequential()
    