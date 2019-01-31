# -*- coding: utf-8 -*-

import keras
import numpy as np
from DataLoaders.IQ_DataLoader import IQ_DataLoader
from Generators.AE_IQ_Generator import AE_IQ_Generator
from Models.Autoencoders import encoder_model_200, brain_enc_model, CNN_3D_AE
from Callbacks.LR_decay import get_callbacks
import nibabel as nib


main_dr = '/home/sage/GenDiagFramework/'
input_dims = (160, 192, 160, 1)
load = False
initial_lr = .0001
num_to_load = None
epochs = 100
preloaded = True
bs = 6
model_path = main_dr + 'saved_models/AE_IQ.h5'


dl = IQ_DataLoader(
                 init_location = '/mnt/sdb1/attempt/Images/training/',
                 label_location = '/home/sage/Neuro/ABCD_Challenge/training_fluid_intelligenceV1.csv',
                 seg_input_size = input_dims,
                 limit=num_to_load,
                 scale_labels=False,
                 in_memory=False,
                 memory_loc='/mnt/sdb1/temp/',
                 compress=False,
                 preloaded=preloaded
                 )

def create_gens(train, test):
    
   gen = AE_IQ_Generator(data_points = train,
                 dim=input_dims,
                 batch_size = bs,
                 n_classes = 1,
                 shuffle = True,
                 augment = False,
                 distort = True,
                 dist_scale = .05,
                 permute = True
                 )

   test_gen = AE_IQ_Generator(data_points = test,
                 dim=input_dims,
                 batch_size = bs,
                 n_classes = 1,
                 shuffle = False,
                 augment = False)
   
   return gen, test_gen

train, test = dl.get_train_test_split(.2, 43)

print(len(train), len(test))

model = CNN_3D_AE(input_dims, 200)
model.compile(loss = 'binary_crossentropy', optimizer = keras.optimizers.adam(initial_lr))

if load:
    model.load_weights(model_path)
    print('loaded weights')

model.summary()
gen, test_gen = create_gens(train, test)

callbacks = get_callbacks(model_file = model_path,
                          initial_learning_rate=initial_lr,
                          learning_rate_drop=.5,
                          learning_rate_epochs=None,
                          learning_rate_patience=100,
                          verbosity=1,
                          early_stopping_patience=100)

model.fit_generator(generator=gen,
                    validation_data=test_gen,
                    use_multiprocessing=True,
                    workers=8,
                    epochs=epochs,
                    callbacks=callbacks)

preds = model.predict_generator(test_gen)


for i in range(5):
    ex = preds[i]
    final = nib.Nifti1Image(ex, np.eye(4))
    final.to_filename(main_dr + 'predictions/test' + str(i) + '.nii.gz')


