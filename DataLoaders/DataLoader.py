#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 11:00:27 2018

@author: sage
"""
from sklearn.model_selection import train_test_split

class DataLoader():
    '''Abstract DataLoader Class'''
    
    def __init__(self,
                 init_location,    #Location of raw nifti files
                 label_location,   #Location of the labels file
                 ):
        
        self.init_location = init_location
        if self.init_location[-1] != '/':
             self.init_location += '/'
             
        self.label_location = label_location
        
        #Init data points
        self.data_points = None
    
    def load_labels(self):
        pass
        
    def load_data(self):
        pass
    
    def get_unique_patients(self):
        
        patients = sorted(list(set([dp.get_patient() for dp in self.data_points])))
        return patients
    
    def get_data_points_by_patient(self, patients):
        
        relevant = []
        
        for dp in self.data_points:
            if dp.get_patient() in patients:
                relevant.append(dp)
        
        return relevant
    
    def load_all(self):
        
        self.load_labels()
        self.load_data()
    
    def get_all(self):
        '''Return just one set, for just train'''
        
        self.load_all()
        
        return self.data_points
    
    def get_train_test_split(self, test_size, seed):
        
        self.load_all()
        patients = self.get_unique_patients()
        
        train_patients, test_patients = train_test_split(patients,
                                                         test_size=test_size,
                                                         random_state=seed)
        
        train = self.get_data_points_by_patient(train_patients)
        test = self.get_data_points_by_patient(test_patients)
        
        return train, test
        
    def get_train_test_val_split(self, test_size, val_size, seed):
        
        self.load_all()
        patients = self.get_unique_patients()
        
        train_patients, test_patients = train_test_split(patients,
                                                         test_size=test_size,
                                                         random_state=seed)
        train_patients, val_patients = train_test_split(train_patients,
                                                        test_size=val_size,
                                                        random_state=seed)
        
        train = self.get_data_points_by_patient(train_patients)
        test = self.get_data_points_by_patient(test_patients)
        val = self.get_data_points_by_patient(val_patients)
        
        return train, test, val
        
    

    
    
    
        