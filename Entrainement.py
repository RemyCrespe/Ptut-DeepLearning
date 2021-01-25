#importation des bibliothèques
import os
import numpy.random

numpy.random.seed(101)

import sys

import pandas as pd
import numpy as np
import cv2
import albumentations as albu

import imageio
import skimage
import skimage.io
import skimage.transform

from skimage.io import imread, imshow
from skimage.transform import resize
from sklearn.utils import shuffle
from sklearn.model_selection import  train_test_split

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
from segmentation_models import get_preprocessing  # this line has an error in the docs

from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

from segmentation_models.losses import dice_loss
# from segmentation_models.metrics import dice_score

from segmentation_models.utils import set_trainable

#déclaration des variables
brain_path = sys.argv[1]#base_path + 'brain_image_dir/'
mask_path = sys.argv[2] #base_path + 'mask_dir/'

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_CHANNELS = 3
BATCH_SIZE = int(sys.argv[3])

#définition des fonctions
def augment_image_and_mask(augmentation, image, mask):
    aug_image_dict = augmentation(image=image, mask=mask)
    image_matrix = aug_image_dict['image']
    mask_matrix = aug_image_dict['mask']
    return image_matrix, mask_matrix


def train_generator(batch_size=10):
    while True:

        # load the data in chunks (batches)
        for df in pd.read_csv('df_train.csv.gz', chunksize=batch_size):

            # get the list of images
            image_id_list = list(df['img'])
            mask_id_list = list(df['mask'])
            #print(image_id_list)

            # Create empty X matrix - 3 channels
            X_train = np.zeros((len(df), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.uint8)

            # create empty Y matrix - 1 channel
            Y_train = np.zeros((len(df), IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.bool)

            # Create X_train
            # ================

            for i in range(0, len(image_id_list)):
                # get the image and mask
                image_id = image_id_list[i]
                mask_id = mask_id_list[i]

                # set the path to the image
                path = brain_path + image_id

                # read the image
                image = cv2.imread(path)

                # convert to from BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # resize the image
                image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))

                # Create Y_train
                # ===============

                # set the path to the mask
                path = mask_path + mask_id

                # read the mask
                mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)

                # resize the mask
                mask = cv2.resize(mask, (IMAGE_HEIGHT, IMAGE_WIDTH))

                # expand dims from (800,600) to (800,600,1)
                mask = np.expand_dims(mask, axis=-1)

                # Augment the image and mask
                # ===========================

                aug_image, aug_mask = augment_image_and_mask(aug_types, image, mask)

                # insert the image into X_train
                X_train[i] = aug_image

                # insert the image into Y_train
                Y_train[i] = aug_mask

            # Normalize the images
            X_train = X_train / 255
            yield X_train, Y_train



def val_generator(batch_size=10):
    while True:

        # load the data in chunks (batches)
        for df in pd.read_csv('df_val.csv.gz', chunksize=batch_size):

            # get the list of images
            image_id_list = list(df['img'])
            mask_id_list = list(df['mask'])

            # Create empty X matrix - 3 channels
            X_val = np.zeros((len(df), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.uint8)

            # create empty Y matrix - 1 channel
            Y_val = np.zeros((len(df), IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.bool)

            # Create X_val
            # ================

            for i, image_id in enumerate(image_id_list):
                # set the path to the image
                path = brain_path + image_id

                # read the image
                image = cv2.imread(path)

                # convert to from BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # resize the image
                image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))

                # insert the image into X_train
                X_val[i] = image

            # Create Y_val
            # ===============

            for j, mask_id in enumerate(mask_id_list):
                # set the path to the mask
                path = mask_path + mask_id

                # read the mask
                mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)

                # resize the mask
                mask = cv2.resize(mask, (IMAGE_HEIGHT, IMAGE_WIDTH))

                # expand dims from (800,600) to (800,600,1)
                mask = np.expand_dims(mask, axis=-1)

                # insert the image into Y_train
                Y_val[j] = mask

            # Normalize the images
            X_val = X_val / 255

            yield X_val, Y_val

aug_types = albu.Compose([
    albu.HorizontalFlip(p=0.5),
    albu.OneOf([
        albu.RandomBrightnessContrast(),
        albu.RandomGamma(),
    ], p=0.3),
    albu.OneOf([
        albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        albu.GridDistortion(),
        albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
    ], p=0.3),
    albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10, shift_limit=0.1, p=0.5, border_mode=0),
])

#Péparation de la base de données
listmimg = os.listdir(brain_path)
listmMask = os.listdir(mask_path)

database = {
    'img' : listmimg,
    'mask' : listmMask
}
print("test", database)

df = pd.DataFrame(database)

df_train, df_val = train_test_split(df, train_size=0.75, random_state=107)
df_train.to_csv('df_train.csv.gz', compression='gzip', index=False)
df_val.to_csv('df_val.csv.gz', compression='gzip', index=False)

#Création du modèle
BACKBONE = 'densenet121'
preprocess_input = get_preprocessing(BACKBONE)

# Note that the model takes 3-channel images as input
model = Unet(BACKBONE, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
             # freeze_encoder=False,
             classes=1,
             encoder_weights='imagenet',
             activation='sigmoid')

# model.summary()
model.compile(
    Adam(lr=0.0001),
     loss=dice_loss,
     metrics=[iou_score],
)

train_gen = train_generator(batch_size=10)

train_steps = int(sys.argv[4])

val_steps = int(sys.argv[5])
epoch = int(sys.argv[6])

train_gen = train_generator(batch_size=BATCH_SIZE)
val_gen = val_generator(batch_size=BATCH_SIZE)



#Entrainement du modèle
filepath = "model.h5"

earlystopper = EarlyStopping(patience=5, verbose=1)

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2,
                              verbose=1, mode='min')

log_fname = 'training_log.csv'
csv_logger = CSVLogger(filename=log_fname,
                       separator=',',
                       append=False)

callbacks_list = [checkpoint, earlystopper, csv_logger, reduce_lr]

history = model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=epoch,
                              validation_data=val_gen, validation_steps=val_steps,
                              verbose=1,
                              callbacks=callbacks_list)
