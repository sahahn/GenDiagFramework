#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 11:09:31 2018

@author: sage
"""
import numpy as np
import gzip

class DataPoint():
    
    def __init__(self,
                 name,
                 label,
                 in_memory,
                 memory_loc=None,
                 slc=None,
                 compress=False
                 ):
        
        self.name = name
        self.label = label
        self.in_memory = in_memory
        self.memory_loc = memory_loc
        self.slc = slc
        self.compress = compress
        
        self.data = None
        self.affine = None
        self.pixdims = None
        
    def get_ref(self):
        return self.name + str(self.slc)
    
    def get_patient(self):
        return self.name.split('_')[0][:-1]
        
    def set_data(self, data):
        
        if self.in_memory:
            self.data = data
            
        elif not self.compress:
            np.save(self.memory_loc + self.get_ref(), data)
            
        else:
            
            f = gzip.GzipFile(self.memory_loc + self.get_ref() + '.npy.gz', 'w')
            np.save(f, data)
            f.close()
            
    def get_data(self, copy=False):
            
        if self.in_memory and not copy:
            return self.data
        
        elif self.in_memory and copy:
            return np.copy(self.data)
            
        elif not self.compress:
            return np.load(self.memory_loc + self.get_ref() + '.npy')

        else:
            
            f = gzip.GzipFile(self.memory_loc + self.get_ref() + '.npy.gz', 'r')
            data = np.load(f)
            f.close()
            
            return data
        
    def clear_data(self):
        
        del self.data
        
    def get_label(self, copy=False):
        
        if copy:
            return np.copy(self.label)
        
        return self.label
    
    def set_affine(self, affine):
        self.affine = affine
        
    def set_pixdims(self, pixdims):
        self.pixdims = pixdims
        
    def get_thickness(self):
        return self.pixdims[3]
        
        
        
        
        
