# -*- coding: utf-8 -*-


import RetinaNet.backbone, RetinaNet.losses, keras
from keras.models import load_model



def RetinaNet_Train(bb_name='resnet50', n_classes=1, optimizer=keras.optimizers.adam):
    

    bb = RetinaNet.backbone.backbone(bb_name)
    model = bb.retinanet(num_classes=n_classes)

    model.compile(
        loss={
            'regression'    : RetinaNet.losses.smooth_l1(),
            'classification': RetinaNet.losses.focal()
        },
        optimizer=optimizer
        )
    
    return model

def load_RN_model(model_loc):
    model = load_model(model_loc,
    custom_objects = RetinaNet.backbone.backbone('resnet50').custom_objects)
    
    return model

