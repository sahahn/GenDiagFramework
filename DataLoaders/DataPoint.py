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
        self.in_memory = in_memory
        self.memory_loc = memory_loc
        self.slc = slc
        self.compress = compress
        
        self.set_label(label)
        
        self.data = None
        self.affine = None
        self.pixdims = None
        self.pred_label = None
        
    def set_data(self, data):
        
        if self.in_memory:
            self.data = data
        
        elif self.compress:
            f = gzip.GzipFile(self.memory_loc + self.get_ref() + '.npy.gz', 'w')
            np.save(f, data)
            f.close()
        
        else:
            np.save(self.memory_loc + self.get_ref(), data)

    def set_label(self, label):
        
        
        if self.in_memory:
            self.label = label
        
        elif self.compress:
            f = gzip.GzipFile(self.memory_loc + self.get_ref() + 'label.npy.gz', 'w')
            np.save(f, label)
            f.close()
        
        else:
            np.save(self.memory_loc + self.get_ref() + 'label', label)
            
    def set_affine(self, affine):
        self.affine = affine
        
    def set_pixdims(self, pixdims):
        self.pixdims = pixdims
        
    def update_dims(self, s_scale, c_scale, a_scale):
        self.pixdims /= [s_scale, c_scale, a_scale]
        
    def set_pred_label(self, pred_label):
        
        if self.in_memory:
            self.pred_label = pred_label
        
        elif self.compress:
            f = gzip.GzipFile(self.memory_loc + self.get_ref() + 'pred_label.npy.gz', 'w')
            np.save(f, pred_label)
            f.close()
        
        else:
            np.save(self.memory_loc + self.get_ref() + 'pred_label', pred_label)
        
        
    def get_ref(self):
        return self.name + str(self.slc)
    
    def get_patient(self):
        return self.name.split('_')[0][:-1]
    
    
    def get_label(self, copy=False):
        
        if self.in_memory and not copy:
            return self.label
        
        elif self.in_memory and copy:
            return np.copy(self.label)
            
        elif not self.compress:
            return np.load(self.memory_loc + self.get_ref() + 'label.npy')

        else:
            
            f = gzip.GzipFile(self.memory_loc + self.get_ref() + 'label.npy.gz', 'r')
            label = np.load(f)
            f.close()
            
            return label
    
    def get_thickness(self):
        return self.pixdims[2]
    
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
        
    def get_pred_label(self, copy=False):
        
        if self.in_memory and not copy:
            return self.pred_label
        
        elif self.in_memory and copy:
            return np.copy(self.pred_label)
            
        elif not self.compress:
            return np.load(self.memory_loc + self.get_ref() + 'pred_label.npy')

        else:
            
            f = gzip.GzipFile(self.memory_loc + self.get_ref() + 'pred_label.npy.gz', 'r')
            label = np.load(f)
            f.close()
            
            return label
        
    
    def get_name(self):
        return self.name
    
    def get_slc(self):
        return self.slc
    
    def get_affine(self, copy=False):
        
        if copy:
            return np.copy(self.affine)
        
        return self.affine
    
    def get_pixdims(self, copy=False):
        
        if copy:
            return np.copy(self.pixdims)
        
        return self.pixdims
        
    def clear_data(self):
        
        del self.data
        

        
        
        
        
        
