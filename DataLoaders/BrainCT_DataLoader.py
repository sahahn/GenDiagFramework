# -*- coding: utf-8 -*-

from DataLoaders.Seg_DataLoader import Seg_DataLoader
from DataUtils.tools import normalize_data
import numpy as np

class BrainCT_DataLoader(Seg_DataLoader):
    '''Overrides Seg_DataLoader'''
    
    def extra_process(self, dp):
        
        data = dp.get_data()
        
        proced = []
        
        for ind in range(self.seg_input_size[0]):
            
            #Set to channels first
            x = data[ind].transpose(2,0,1)
            x = np.clip(x, *self.clip_range[ind])
            x = normalize_data(x)
            
            fill = np.zeros(self.seg_input_size[1:])
            dif = self.seg_input_size[1] - len(x)
            
            for i in range(len(x)):
                fill[(dif // 2)+i] = x[i]
                
            proced.append(fill)
            
        dp.set_data(np.array(proced))
            
        label = dp.get_label().transpose(2,0,1)
        
        fill = np.zeros(self.seg_input_size[1:])
        dif = self.seg_input_size[1] - len(label)
        
        for i in range(len(label)):
            fill[(dif // 2)+i] = label[i]
            
        dp.set_label(np.expand_dims(fill, axis=0))
        
        return dp
