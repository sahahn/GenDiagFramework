# -*- coding: utf-8 -*-
import numpy as np
import RetinaNet.backbone, RetinaNet.losses, keras
from RetinaNet.retinanet import retinanet_bbox

from RetinaNet.utils.anchors import anchors_for_shape, anchor_targets_bbox

num_classes = 1

bb = RetinaNet.backbone.backbone('resnet50')
model = bb.retinanet(num_classes)

pred_model = retinanet_bbox(model)


model.compile(
        loss={
            'regression'    : RetinaNet.losses.smooth_l1(),
            'classification': RetinaNet.losses.focal()
        },
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    )

model.summary()

num_fake = 10

image_shape = (512,512,3)
images = np.ones((num_fake, *image_shape))

annotations = np.ones((num_fake, 5))
annotations[:,0:2]=1
annotations[:,2:4]=100

annotations[0] = np.empty((5,))


anchors = anchors_for_shape(image_shape)

targets = list(anchor_targets_bbox(anchors, images, annotations, 1))



for t in targets:
    print(np.shape(t))

model.fit(x=images, y=targets)

predictions = model.predict(images)

#annotations['bboxes'][:, 0] #x1
#annotations['bboxes'][:, 1] #y1
#annotations['bboxes'][:, 2] #x2
#annotations['bboxes'][:, 3] #y2
#annotations['labels'] = 0