# -*- coding: utf-8 -*-

from DataLoaders.RN_DataLoader import RN_DataLoader
from Generators.RN_Generator import RN_Generator

import RetinaNet.backbone, RetinaNet.losses, keras
from RetinaNet.retinanet import get_predictions
from config import config
from keras.models import load_model

import numpy as np

data_loader = RN_DataLoader(
        init_location = '/media/sage/data/nifti_endoleak/',
        label_location = '/home/sage/GenDiagFramework/labels/annotations.csv')
        #label_location = '/home/sage/small.csv')

train, test = data_loader.get_train_test_split(.2, 43)
np.warnings.filterwarnings('ignore')

print(len(train), len(test))

train_gen = RN_Generator(data_points = train,
                             dim=(512, 512, 1),
                             batch_size = 1,
                             n_classes = 1,
                             shuffle = True,
                             augment = True,
                             label_size = 5)

test_gen = RN_Generator(data_points = test,
                             dim=(512, 512, 1),
                             batch_size = 1,
                             n_classes = 1,
                             shuffle = False,
                             augment = False,
                             label_size = 5)


bb = RetinaNet.backbone.backbone('resnet50')
model = bb.retinanet(num_classes=1)

model.compile(
        loss={
            'regression'    : RetinaNet.losses.smooth_l1(),
            'classification': RetinaNet.losses.focal()
        },
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    )

#model = load_model(config['model_loc'] + '/model-01.h5',
#                   custom_objects = RetinaNet.backbone.backbone('resnet50').custom_objects)

callbacks =  [keras.callbacks.ModelCheckpoint(config['model_loc'] + '/model-{epoch:02d}.h5')]

model.fit_generator(generator=train_gen,
                            use_multiprocessing=True,
                            workers=8,
                            epochs=1,
                            callbacks=callbacks)

boxes, scores = get_predictions(model, test_gen, .1)

for i in range(len(boxes)):
    print(boxes[i], scores[i])