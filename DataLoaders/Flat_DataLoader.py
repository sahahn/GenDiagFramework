from DataLoaders.IQ_DataLoader import IQ_DataLoader
import os
import numpy as np

def load_map(dr, name, ind):
    
    left = np.load(dr + name + '-lh.npy')[:,:,ind]
    right = np.load(dr + name + '-rh.npy')[:,:,ind]
    
    full = np.stack([left, right], axis=-1)
    return full

class Flat_DataLoader(IQ_DataLoader):
    ''' Input size refers to inds to select because Im a bad person'''

    def load_data(self):
        
        all_names = os.listdir(self.init_location)
        all_names = [name for name in all_names if 'NDAR' in name]
        
        names = [name.split('-')[0] for name in all_names]
        names = list(set([name for name in names if name + '-rh.npy' in all_names and
                                           name + '-lh.npy' in all_names and
                                           name in self.iq_dict]))

        map_inds = self.input_size

        if self.preloaded == False:

            s_norm_info = []

            for ind in map_inds:
                data = np.array([load_map(self.init_location, name, ind) for name in names[:500]])
                
                data_mean = np.mean(data)
                data -= data_mean
                data_max = np.max(data)
                data /= data_max
                imax = np.max(data)
                imin = np.min(data)

                s_norm_info.append([data_mean, data_max, imax, imin])

        for name in names:
            if (len(self.data_points) < self.limit) and (name in self.iq_dict):
            
                label = self.iq_dict[name]
                dp = self.create_data_point(name, label)
            
                if self.preloaded == False:

                    all_maps = []

                    for i in range(len(map_inds)):
                        data = load_map(self.init_location, name, map_inds[i])

                        data = data.astype('float32')
                        data -= s_norm_info[i][0]
                        data /= s_norm_info[i][1]
                        data -= s_norm_info[i][3]
                        data /= (s_norm_info[i][2] - s_norm_info[i][3])

                        all_maps.append(data)

                    all_maps = np.concatenate(all_maps, axis=-1)
                    dp.set_data(all_maps)
                
                self.data_points.append(dp)

