import numpy.random
numpy.random.seed(101)

import sys

from PIL import Image

import pandas as pd
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, UpSampling2D
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

from keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint, CSVLogger, LearningRateScheduler)


from keras.optimizers import Adam
from keras.losses import binary_crossentropy

from keras.initializers import he_normal

import tensorflow as tf

from segmentation_models import Unet, FPN
from segmentation_models import  get_preprocessing # this line has an error in the docs

from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

from segmentation_models.losses import dice_loss
#from segmentation_models.metrics import dice_score
import matplotlib.pyplot as plt
import cv2

from segmentation_models.utils import set_trainable

BACKBONE = 'densenet121'
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_CHANNELS = 3

model = Unet(BACKBONE, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
             #freeze_encoder=False,
             classes=1,
             encoder_weights='imagenet',
             activation='sigmoid')

#model.summary()


model.load_weights('model.h5')

#X_test, Y_test = next(test_gen)
X_test = np.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.uint8)

Y_test = np.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.bool)

img = sys.argv[1]
#path = 'brain_image_dir/' + '52_12'+'.jpg'
path = img

#test_image = Image.open('brain_image_dir/71_1.jpg')
#test_image.show()

# read the image
image = cv2.imread(path)

# convert to from BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# resize the image
image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))

# insert the image into X_train
X_test[0] = image

#path = 'mask_dir/' + '52_12'+'_HGE_Seg.jpg'
path = 'D:\\ptut\\Ptut-DeepLearning\\mask_dir\\50_12_HGE_Seg.jpg'
# read the mask
mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)

# resize the mask
mask = cv2.resize(mask, (IMAGE_HEIGHT, IMAGE_WIDTH))

# expand dims from (800,600) to (800,600,1)
mask = np.expand_dims(mask, axis=-1)

# insert the image into Y_train
Y_test[0] = mask

# Normalize the images
X_test = X_test / 255

test_data = (X_test , Y_test)

prediction = model.predict(X_test, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10,
    workers=1, use_multiprocessing=False
)

#test_data = (X_test , Y_test)
print(prediction.min())
print(prediction.max())

preds_test_thresh = (prediction >= 0.7).astype(np.uint8)

preds_test_thresh.shape

print(preds_test_thresh.min())
print(preds_test_thresh.max())



predict_mask1 = preds_test_thresh[0,:,:,0]
#plt.imshow(predict_mask1, cmap='Reds', alpha=0.3)
#plt.show()

#img = Image.fromarray(image)
#img = Image.fromarray(predict_mask1,cmap='Reds', alpha=0.3)
#img = Image.fromarray(predict_mask1)
#img.save('test.jpg', cmap='Reds', alpha=0.3)

plt.imshow(image, cmap='gray')
plt.imshow(predict_mask1, cmap='Reds', alpha=0.3)
axes = plt.axes()
axes.get_xaxis().set_visible(False)
axes.get_yaxis().set_visible(False)
#plt.axes('off')

plt.savefig('temp.jpg', bbox_inches='tight', pad_inches=0)
#plt.show()



predict_mask = prediction[0,:,:,0]
plt.imshow(predict_mask, cmap='Reds', alpha=0.3)
#plt.show()

true_mask = Y_test[0,:,:,0]
plt.imshow(true_mask, cmap='Blues', alpha=0.3)
#plt.show()

image = X_test[0,:,:,:]
plt.imshow(image)
#plt.show()

plt.imshow(image, cmap='gray')

plt.imshow(true_mask, cmap='Reds', alpha=0.3)
#plt.show()

#plt.imshow(mask, cmap='Blues', alpha=0.3)
#plt.show()
