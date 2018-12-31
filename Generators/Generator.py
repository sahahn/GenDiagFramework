#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 14:45:56 2018

@author: sage
"""

import keras
import numpy as np

class Generator(keras.utils.Sequence):
    
    def __init__(self,
                 data_points,
                 dim,
                 batch_size,
                 n_classes,
                 shuffle,
                 augment,
                 label_size = None):
    
        self.data_points = data_points
        self.dim = dim
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        
        if label_size == None:
            label_size = self.n_classes
            
        self.label_size = label_size
        
        #Create an internal list of IDs which simply index data points
        self.list_IDs = np.arange(len(data_points))
        
        #Inits indexes - an initial shuffle if shuffle
        self.on_epoch_end()
                 
    def on_epoch_end(self):
 
        self.indexes = np.arange(len(self.list_IDs))
    
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    
    def __getitem__(self, index):

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.data_generation(list_IDs_temp)
        
        #Apply extra functions, e.g. for Retina Net signifigant changes
        X, y = self.extra(X, y)

        return X, y
    
    
    def data_generation(self, list_IDs_temp):

        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, self.label_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            X[i], y[i] = self.get_sample_from(ID)
       
        return X, y
    
    def get_sample_from(self, ID):
        
        #If augment and in_memory return copies
        x = self.data_points[ID].get_data(copy=self.augment)
        y = self.data_points[ID].get_label(copy=self.augment)
        
        if self.augment:
            x,y = self.data_aug(x, y)
            
        return x, y
    
    def data_aug(self, x, y):
        return x, y
        
    def extra(self, X, y):
        return X, y
    



        
        










    
    
   