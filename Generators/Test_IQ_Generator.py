from Generators.Seg_Generator import Seg_Generator
import numpy as np

class Test_IQ_Generator(Seg_Generator):

    def __init__(self,
                 data_points,
                 dim,
                 batch_size,
                 n_classes,
                 label_size = None,
                 to_remove = None):

        super().__init__(data_points, dim, batch_size, n_classes,
             shuffle = False, augment = False, label_size = label_size)

        self.to_remove = to_remove

    def get_sample_from(self, ID):
        
        #If augment return a copy
        x = self.data_points[ID].get_data(copy=True)
        y = self.data_points[ID].get_label(copy=True)

        if self.to_remove != None:

            seg = self.data_points[ID].get_guide_label()

            for i in to_remove:
                x[seg == i] = 0

        return x, y
            
    def data_generation(self, list_IDs_temp):

        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, self.n_classes))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            X[i], y[i] = self.get_sample_from(ID)
       
        return X, y