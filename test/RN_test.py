
# -*- coding: utf-8 -*-

from DataLoaders.RN_DataLoader import RN_DataLoader
from Generators.RN_Generator import RN_Generator

from Models.RetinaNet import RetinaNet_Train, load_RN_model
import keras
from RetinaNet.retinanet import get_predictions
from config import config

import numpy as np

data_loader = RN_DataLoader(
        init_location = '/mnt/sdb2/data/nifti_endoleak/',
        label_location = '/home/sage/GenDiagFramework/labels/annotations.csv',
        in_memory = False,
        memory_loc = '/mnt/sda5/temp/',
        compress = True,
        preloaded=True)

train, test = data_loader.get_train_test_split(.75, 43)
np.warnings.filterwarnings('ignore')

print(len(train), len(test))

train_gen = RN_Generator(data_points = train,
                             dim=config['RN_input_size'],
                             batch_size = 6,
                             n_classes = 1,
                             shuffle = True,
                             augment = True,
                             label_size = 5) 

test_gen = RN_Generator(data_points = test,
                             dim=config['RN_input_size'],
                             batch_size = 1,
                             n_classes = 1,
                             shuffle = False,
                             augment = False,
                             label_size = 5)


model = load_RN_model('/home/sage/GenDiagFramework/saved_models/RN-06.h5')


#model = RetinaNet_Train()
callbacks =  [keras.callbacks.ModelCheckpoint(config['model_loc'] + '/RN-{epoch:02d}.h5')]

model.fit_generator(generator=train_gen,
                            validation_data=test_gen,
                            use_multiprocessing=True,
                            workers=8,
                            epochs=25,
                            callbacks=callbacks)



'''
boxes, scores = get_predictions(model, test_gen, .1)

for i in range(len(boxes)):
    test[i].set_pred_label(boxes[i])
    
    print(test[i].pred_label)
    '''
