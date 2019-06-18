#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 11:00:27 2018

@author: sage
"""
from sklearn.model_selection import train_test_split
from DataLoaders.DataPoint import DataPoint
from sklearn.model_selection import KFold
import numpy as np
import os

class DataLoader():
    '''Abstract DataLoader Class'''
    
    def __init__(self,
                 init_location,    
                 label_location,
                 in_memory=True,
                 memory_loc=None,
                 compress=False,
                 preloaded=False,
                 ):
        
        ''' 
        init_location - Location of the raw data files.
        label_location - Location of the label file/folder.
        in_memory - Flag to determine if the data should be loaded into memory
                    or if it should be saved to disk.
        memory_loc - If saved to disk, location to save to.
        compress - Flag to determine if the data should be saved in a
                   compressed form.
        preloaded - Flag to indicate if data has already been saved in temp mem spot
        '''
        
        self.init_location = os.path.abspath(init_location)
        self.label_location = os.path.abspath(label_location)
        
        self.in_memory = in_memory
        self.memory_loc = memory_loc
        self.compress = compress
        self.preloaded = preloaded
    
        self.data_points = []
        
        self.kf_train_splits = []
        self.kf_test_splits = []
    
    def load_labels(self):
        pass
        
    def load_data(self):
        pass
    
    def load_new(self):
        pass
    
    def load_unseen(self):
        self.load_new()
        
        return self.data_points
        
    
    def create_data_point(self, name, label, slc=None):

        if type(label) == list:

            dp = DataPoint(
                name = name,
                label = label[0],
                extra = label[1:],
                in_memory = self.in_memory,
                memory_loc = self.memory_loc,
                compress = self.compress,
                slc = slc )

        else:

            dp = DataPoint(
                    name = name,
                    label = label,
                    in_memory = self.in_memory,
                    memory_loc = self.memory_loc,
                    compress = self.compress,
                    slc = slc )
        
        return dp
        
    
    def get_unique_patients(self):
        
        patients = sorted(list(set([dp.get_patient() for dp in self.data_points])))
        return np.array(patients)
    
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
    
    def setup_kfold_splits(self, n_splits, seed):
        '''Preforms a kfold val_split by patient, and stores the splits.'''
        
        self.load_all()
        patients = self.get_unique_patients()
        
        kf = KFold(n_splits=n_splits, random_state=seed)
        
        for train_patients, test_patients in kf.split(patients):
            self.kf_train_splits.append(patients[train_patients])
            self.kf_test_splits.append(patients[test_patients])
            

    def get_k_split(self, ind):
        
        train_patients = self.kf_train_splits[ind]
        test_patients = self.kf_test_splits[ind]
        
        train = self.get_data_points_by_patient(train_patients)
        test = self.get_data_points_by_patient(test_patients)
        
        return train, test
        
    

    
    
    
        
