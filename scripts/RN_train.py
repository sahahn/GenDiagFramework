# -*- coding: utf-8 -*-


from DataLoaders.RN_DataLoader import RN_DataLoader
from Generators.RN_Generator import RN_Generator

from Models.RetinaNet import RetinaNet_Train, load_RN_model
import keras
from RetinaNet.retinanet import get_predictions
from config import config
from Callbacks.LR_decay import get_callbacks
import numpy as np

main_dr = '/home/sage/GenDiagFramework/'
initial_lr =1e-5 
bs = 6

np.warnings.filterwarnings('ignore')

data_loader = RN_DataLoader(
        init_location = '/mnt/sdb2/data/nifti_endoleak/',
        label_location = '/home/sage/GenDiagFramework/labels/annotations.csv',
        in_memory = False,
        memory_loc = '/mnt/sda5/temp/',
        compress = True,
        preloaded = True)

train, test, val = data_loader. get_train_test_val_split(.2, .3, 43)
print(len(train), len(val))

train_gen = RN_Generator(data_points = train,
                             dim=config['RN_input_size'],
                             batch_size = bs,
                             n_classes = 1,
                             shuffle = True,
                             augment = True,
                             label_size = 5) 

test_gen = RN_Generator(data_points = val,
                             dim=config['RN_input_size'],
                             batch_size = bs,
                             n_classes = 1,
                             shuffle = False,
                             augment = False,
                             label_size = 5)

callbacks = get_callbacks(model_file = main_dr + 'saved_models/RN.h5',
                          initial_learning_rate = initial_lr,
                          learning_rate_drop=.5,
                          learning_rate_epochs=None,
                          learning_rate_patience=10,
                          verbosity=1,
                          early_stopping_patience=50)


#model = load_RN_model('/home/sage/GenDiagFramework/saved_models/RN.h5')

model = RetinaNet_Train()
model.fit_generator(generator=train_gen,
                            validation_data=test_gen,
                            use_multiprocessing=True,
                            workers=8,
                            epochs=25,
                            callbacks=callbacks)


