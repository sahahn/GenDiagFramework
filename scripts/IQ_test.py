# -*- coding: utf-8 -*-


import keras
import numpy as np
from DataLoaders.IQ_DataLoader import IQ_DataLoader
from Generators.IQ_Generator import IQ_Generator
from Models.Resnet3D import Resnet3DBuilder
from Callbacks.LR_decay import get_callbacks
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error



input_dims = (256, 256, 256, 1)
scale_labels = True
initial_lr = .001
num_to_load = 600
epochs = 100
main_dr = '/home/sage/GenDiagFramework/'

def create_gens(train, test):
    
   gen = IQ_Generator(data_points = train,
                 dim=input_dims,
                 batch_size = 1,
                 n_classes = 1,
                 shuffle = True,
                 augment = False,
                 distort = False,
                 dist_scale = .1,
                 permute = False
                 )

   test_gen = IQ_Generator(data_points = test,
                 dim=input_dims,
                 batch_size = 1,
                 n_classes = 1,
                 shuffle = False,
                 augment = False)
   
   return gen, test_gen


dl = IQ_DataLoader(
                 init_location = '/media/sage/Images/training/',    
                 label_location = '/home/sage/Neuro/ABCD_Challenge/training_fluid_intelligenceV1.csv',
                 seg_input_size = input_dims,
                 limit=num_to_load,
                 scale_labels=scale_labels,
                 in_memory=False,
                 memory_loc='/mnt/sda5/temp/',
                 compress=False,
                 preloaded=False
                 )

train, test = dl.get_train_test_split(.2, 43)

print(len(train), len(test))

rn_builder = Resnet3DBuilder()
model = rn_builder.build_resnet_18(input_shape=input_dims, num_outputs=1, reg_factor=1e-6)
model.compile(loss = 'mean_squared_error', optimizer = keras.optimizers.sgd(initial_lr))

gen, test_gen = create_gens(train, test)

callbacks = get_callbacks(model_file = main_dr + 'saved_models/IQ.h5',
                          initial_learning_rate=initial_lr,
                          learning_rate_drop=.5,
                          learning_rate_epochs=None,
                          learning_rate_patience=10,
                          verbosity=1,
                          early_stopping_patience=50)

model.fit_generator(generator=gen,
                    validation_data=test_gen,
                    use_multiprocessing=True,
                    workers=8,
                    epochs=epochs)

preds = model.predict_generator(test_gen)

for p in range(len(preds)):
    test[p].set_pred_label(float(preds[p]))
    
true = [dp.get_label() for dp in test]
pred = [dp.get_pred_label() for dp in test]

print('True: ')
print(true)
print('Pred: ')
print(pred)

if scale_labels:
 
    dl.reverse_label_scaling()
    
    true = [dp.get_label() for dp in test]
    pred = [dp.get_pred_label() for dp in test]
    
    print('True post reverse: ')
    print(true)
    print('Pred post reverse: ')
    print(pred)

r2_score = r2_score(true, pred)
print('r2 score: ', r2_score)

mae_score = mean_absolute_error(true, pred)
print('mae score: ', mae_score)

mse_score = mean_squared_error(true, pred)
print('mse score: ', mse_score)

                            
                            
