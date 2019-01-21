# -*- coding: utf-8 -*-


import keras
import numpy as np
from DataLoaders.IQ_DataLoader import IQ_DataLoader
from Generators.IQ_Generator import IQ_Generator
from Models.Resnet3D import Resnet3DBuilder
from Models.CNN_3D import CNN_3D
from Callbacks.LR_decay import get_callbacks
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


load = False
input_dims = (240, 240, 240, 1)
scale_labels = False
initial_lr = .00001
num_to_load = None
epochs = 100
main_dr = '/home/sage/GenDiagFramework/'
preloaded = True

def create_gens(train, test):
    
   gen = IQ_Generator(data_points = train,
                 dim=input_dims,
                 batch_size = 2,
                 n_classes = 1,
                 shuffle = True,
                 augment = False,
                 distort = True,
                 dist_scale = .05,
                 permute = True
                 )

   test_gen = IQ_Generator(data_points = test,
                 dim=input_dims,
                 batch_size = 2,
                 n_classes = 1,
                 shuffle = False,
                 augment = False)
   
   return gen, test_gen


dl = IQ_DataLoader(
                 #init_location = '/media/sage/Images/training/',
                 init_location = '/mnt/sdb1/attempt/Images/training/',
                 label_location = '/home/sage/Neuro/ABCD_Challenge/training_fluid_intelligenceV1.csv',
                 seg_input_size = input_dims,
                 limit=num_to_load,
                 scale_labels=scale_labels,
                 in_memory=False,
                 #memory_loc='/mnt/sda5/temp/',
                 memory_loc='/mnt/sdb1/temp/',
                 compress=False,
                 preloaded=preloaded
                 )

train, test = dl.get_train_test_split(.2, 43)

print(len(train), len(test))

#rn_builder = Resnet3DBuilder()
#model = rn_builder.build_resnet_50(input_shape=input_dims, num_outputs=1, reg_factor=1e-6)
model = CNN_3D(input_dims)
model.compile(loss = 'mean_squared_error', optimizer = keras.optimizers.adam(initial_lr))

if load:
    model.load_weights(main_dr + 'saved_models/IQ.h5')
    print('loaded weights')

model.summary()
gen, test_gen = create_gens(train, test)

callbacks = get_callbacks(model_file = main_dr + 'saved_models/IQ.h5',
                          initial_learning_rate=initial_lr,
                          learning_rate_drop=.5,
                          learning_rate_epochs=None,
                          learning_rate_patience=50,
                          verbosity=1,
                          early_stopping_patience=130)

model.fit_generator(generator=gen,
                    validation_data=test_gen,
                    use_multiprocessing=True,
                    workers=8,
                    epochs=epochs,
                    callbacks=callbacks)

preds = model.predict_generator(test_gen)

for p in range(len(preds)):
    test[p].set_pred_label(float(preds[p]))
    
true = [dp.get_label() for dp in test]
pred = [dp.get_pred_label() for dp in test]

#print('True: ')
#print(true)
#print('Pred: ')
#print(pred)

if scale_labels:
 
    dl.reverse_label_scaling()
    
    true = [dp.get_label() for dp in test]
    pred = [dp.get_pred_label() for dp in test]
    
#    print('True post reverse: ')
#    print(true)
#    print('Pred post reverse: ')
#    print(pred)

r2_score = r2_score(true, pred)
print('r2 score: ', r2_score)

mae_score = mean_absolute_error(true, pred)
print('mae score: ', mae_score)

mse_score = mean_squared_error(true, pred)
print('mse score: ', mse_score)

                            
                            
