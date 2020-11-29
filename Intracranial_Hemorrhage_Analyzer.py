import numpy.random
numpy.random.seed(101)

import pandas as pd
import numpy as np
from tkinter import *

#from segmentation_models import  get_preprocessing

import os
import cv2

import imageio
import skimage
import skimage.io
import skimage.transform

from skimage.io import imread, imshow
from skimage.transform import resize

from sklearn.utils import shuffle
from sklearn.model_selection import  train_test_split
import itertools
import shutil
import matplotlib.pyplot as plt
import albumentations as albu


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

from segmentation_models.utils import set_trainable


def augment_image_and_mask(augmentation, image, mask) :
    aug_image_dict = augmentation(image=image, mask=mask)
    image_matrix = aug_image_dict['image']
    mask_matrix = aug_image_dict['mask']
    return image_matrix, mask_matrix

def swap_target(x) :
    if x == 0 :
        return 1
    else:
        return 0

def get_mask_fname(row):
    mask_id = str(row['SliceNumber']) + '_HGE_Seg.jpg'
    return mask_id


def new_mask_fname(row):
    mask_id = str(row['PatientNumber']) + '_' + str(row['SliceNumber']) + '_HGE_Seg.jpg'
    return mask_id


def assign_image_fname(row):
    image_fname = str(row['SliceNumber']) + '.jpg'

    return image_fname


def assign_new_fname(row):
    mask_id = str(row['PatientNumber']) + '_' + str(row['SliceNumber']) + '.jpg'

    return mask_id


def draw_category_images(col_name, figure_cols, df, IMAGE_PATH):

    categories = (df.groupby([col_name])[col_name].nunique()).index
    f, ax = plt.subplots(nrows=len(categories), ncols=figure_cols,
                         figsize=(4 * figure_cols, 4 * len(categories)))  # adjust size here

    # draw a number of images for each location
    for i, cat in enumerate(categories):
        sample = df[df[col_name] == cat].sample(figure_cols)  # figure_cols is also the sample size
        for j in range(0, figure_cols):
            file = IMAGE_PATH + sample.iloc[j]['new_image_fname']
            im = imageio.imread(file)
            ax[i, j].imshow(im, resample=True, cmap='gray')
            ax[i, j].set_title(cat, fontsize=14)

    plt.tight_layout()
    plt.show()


def draw_category_masks(col_name, figure_cols, df, IMAGE_PATH):

    categories = (df.groupby([col_name])[col_name].nunique()).index
    f, ax = plt.subplots(nrows=len(categories), ncols=figure_cols,
                         figsize=(4 * figure_cols, 4 * len(categories)))  # adjust size here
    # draw a number of images for each location
    for i, cat in enumerate(categories):
        sample = df[df[col_name] == cat].sample(figure_cols)  # figure_cols is also the sample size
        for j in range(0, figure_cols):
            file = IMAGE_PATH + sample.iloc[j]['new_mask_fname']
            im = imageio.imread(file)
            ax[i, j].imshow(im, resample=True, cmap='gray')
            ax[i, j].set_title(cat, fontsize=14)
    plt.tight_layout()
    plt.show()


def train_generator(batch_size=10):
    while True:

        # load the data in chunks (batches)
        for df in pd.read_csv('df_train.csv.gz', chunksize=batch_size):

            # get the list of images
            image_id_list = list(df['new_image_fname'])
            mask_id_list = list(df['new_mask_fname'])

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
                path = 'brain_image_dir/' + image_id

                # read the image
                image = cv2.imread(path)

                # convert to from BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # resize the image
                image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))

                # Create Y_train
                # ===============

                # set the path to the mask
                path = 'mask_dir/' + mask_id

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
            image_id_list = list(df['new_image_fname'])
            mask_id_list = list(df['new_mask_fname'])

            # Create empty X matrix - 3 channels
            X_val = np.zeros((len(df), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.uint8)

            # create empty Y matrix - 1 channel
            Y_val = np.zeros((len(df), IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.bool)

            # Create X_val
            # ================

            for i, image_id in enumerate(image_id_list):
                # set the path to the image
                path = 'brain_image_dir/' + image_id

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
                path = 'mask_dir/' + mask_id

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


def test_generator(batch_size=1):
    while True:

        # load the data in chunks (batches)
        for df in pd.read_csv('df_test.csv.gz', chunksize=batch_size):

            # get the list of images
            image_id_list = list(df['new_image_fname'])
            mask_id_list = list(df['new_mask_fname'])

            # Create empty X matrix - 3 channels
            X_test = np.zeros((len(df), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.uint8)

            # create empty Y matrix - 1 channel
            Y_test = np.zeros((len(df), IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.bool)

            # Create X_test
            # ================

            for i, image_id in enumerate(image_id_list):
                # set the path to the image
                path = 'brain_image_dir/' + image_id

                # read the image
                image = cv2.imread(path)

                # convert to from BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # resize the image
                image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))

                # insert the image into X_train
                X_test[i] = image

            # Create Y_test
            # ===============

            for j, mask_id in enumerate(mask_id_list):
                # set the path to the mask
                path = 'mask_dir/' + mask_id

                # read the mask
                mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)

                # resize the mask
                mask = cv2.resize(mask, (IMAGE_HEIGHT, IMAGE_WIDTH))

                # expand dims from (800,600) to (800,600,1)
                mask = np.expand_dims(mask, axis=-1)

                # insert the image into Y_train
                Y_test[j] = mask

            # Normalize the images
            X_test = X_test / 255

            yield X_test, Y_test


base_path = 'D:/ptut/Ptut-DeepLearning/computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0/'

os.listdir(base_path)
print(os.listdir(base_path))

IMAGE_HEIGHT_ORIG = 650
IMAGE_WIDTH_ORIG = 650

NUM_TEST_IMAGES = 10 # 10 with intracranial hem + 10 without intracranial hem

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_CHANNELS = 3

BATCH_SIZE = 10

path = base_path + 'hemorrhage_diagnosis.csv'
df_diag = pd.read_csv(path)

df_diag['Has_Hemorrhage'] = df_diag['No_Hemorrhage'].apply(swap_target)

df_diag = df_diag.drop('No_Hemorrhage', axis=1)

df_diag['mask_fname'] = df_diag.apply(get_mask_fname, axis=1)

df_diag['new_mask_fname'] = df_diag.apply(new_mask_fname, axis=1)

df_diag['image_fname'] = df_diag.apply(assign_image_fname, axis=1)

df_diag['new_image_fname'] = df_diag.apply(assign_new_fname, axis=1)

df_diag.head()

index_to_drop = df_diag[(df_diag['PatientNumber'] == 84) & (df_diag['SliceNumber'] == 36)].index

index_to_drop = index_to_drop[0]

df_diag = df_diag.drop(index_to_drop, axis=0)
df_diag[df_diag.index == index_to_drop]

print(df_diag[df_diag.index == index_to_drop])

print(df_diag['Has_Hemorrhage'].value_counts())
print(df_diag['PatientNumber'].nunique())

path = base_path + 'Patients_CT'
folder_list = os.listdir(path)
print(len(folder_list))

print(len(os.listdir('mask_dir')))
print(len(os.listdir('mask_dir')))
print(len(os.listdir('brain_image_dir')))
print(len(os.listdir('bone_image_dir')))

index = 14
fname =df_diag.loc[index, 'new_image_fname']
path = 'brain_image_dir/' + fname
brain_image = plt.imread(path)

fname = df_diag.loc[index, 'new_mask_fname']
path = 'mask_dir/' + fname
# read the image as a matrix
mask = plt.imread(path)

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

index = 14
fname = df_diag.loc[index, 'new_image_fname']
path = 'brain_image_dir/' + fname
# read the image as a matrix
brain_image = cv2.imread(path)

print(brain_image.shape)

fname = df_diag.loc[index, 'new_mask_fname']
path = 'mask_dir/' + fname
# read the image as a matrix
mask = plt.imread(path)

print(mask.shape)
aug_image, aug_mask = augment_image_and_mask(aug_types, brain_image, mask)

#plt.imshow(aug_image, cmap='gray')
#plt.imshow(aug_mask, cmap='Blues', alpha=0.7)

IMAGE_PATH = 'mask_dir/'

draw_category_masks('Has_Hemorrhage',4, df_diag, IMAGE_PATH)


NUM_TEST_IMAGES = 10

df = df_diag[df_diag['Has_Hemorrhage'] == 0]

df_no_hem = df.sample(NUM_TEST_IMAGES, random_state=101)

df_no_hem = df_no_hem.reset_index(drop=True)

test_images_list = list(df_no_hem['new_mask_fname'])

df_diag = df_diag[~df_diag['new_mask_fname'].isin(test_images_list)]

df = df_diag[df_diag['Has_Hemorrhage'] == 1]

df_with_hem = df.sample(NUM_TEST_IMAGES, random_state=102)

df_with_hem = df_with_hem.reset_index(drop=True)

test_images_list = list(df_with_hem['new_mask_fname'])


df_diag = df_diag[~df_diag['new_mask_fname'].isin(test_images_list)]


df_test = pd.concat([df_with_hem, df_no_hem], axis=0).reset_index(drop=True)

print(df_diag.shape)
print(df_test.shape)


# train_test_split


# shuffle
df_diag = shuffle(df_diag)

# reset the index
df_diag = df_diag.reset_index(drop=True)

# We will stratify by target
y = df_diag['Has_Hemorrhage']

df_train, df_val = train_test_split(df_diag, test_size=0.15, random_state=107, stratify=y)

print(df_train.shape)
print(df_val.shape)
print(df_train['Has_Hemorrhage'].value_counts())
print(df_val['Has_Hemorrhage'].value_counts())

df_diag.to_csv('df_data.csv.gz', compression='gzip', index=False)

df_train.to_csv('df_train.csv.gz', compression='gzip', index=False)
df_val.to_csv('df_val.csv.gz', compression='gzip', index=False)

df_test.to_csv('df_test.csv.gz', compression='gzip', index=False)

#BACKBONE = 'densenet121'
#preprocess_input = get_preprocessing(BACKBONE)

train_gen = train_generator(batch_size=10)

# run the generator
X_train, Y_train = next(train_gen)

print(X_train.shape)
print(Y_train.shape)

img = X_train[7,:,:,:]

msk = Y_train[7,:,:,0]

plt.imshow(img, cmap='gray')
plt.imshow(msk, cmap='Blues', alpha=0.7)


val_gen = val_generator(batch_size=10)

# run the generator
X_val, Y_val = next(val_gen)

print(X_val.shape)
print(Y_val.shape)

msk = Y_val[7,:,:,0]
#plt.imshow(msk)

plt.imshow(img, cmap='gray')
plt.imshow(msk, cmap='Blues', alpha=0.7)

test_gen = test_generator(batch_size=15)

# run the generator
X_test, Y_test = next(test_gen)

print(X_test.shape)
print(Y_test.shape)

img = X_test[14,:,:,:]
plt.imshow(img)
msk = Y_test[14,:,:,0]
plt.imshow(msk)
plt.imshow(img, cmap='gray')
plt.imshow(msk, cmap='Blues', alpha=0.7)

plt.show()

#preprocess = get_preprocessing('resnet101') # for resnet, img = (img-110.0)/1.0

BACKBONE = 'densenet121'
preprocess_input = get_preprocessing(BACKBONE)

# Note that the model takes 3-channel images as input
model = Unet(BACKBONE, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
             #freeze_encoder=False,
             classes=1,
             encoder_weights='imagenet',
             activation='sigmoid')

#model.summary()

test_gen = test_generator(batch_size=len(df_test))

# run the generator
X_test, Y_test = next(test_gen)

print(X_test.shape)
print(Y_test.shape)

#num_train_samples = len(df_train)

num_train_samples = 5
#num_val_samples = len(df_val)
num_val_samples = 5
train_batch_size = BATCH_SIZE
val_batch_size = BATCH_SIZE

# determine numtrain steps
train_steps = np.ceil(num_train_samples / train_batch_size)
# determine num val steps
val_steps = np.ceil(num_val_samples / val_batch_size)

train_gen = train_generator(batch_size=BATCH_SIZE)
val_gen = val_generator(batch_size=BATCH_SIZE)

model.compile(
    Adam(lr=0.0001),
    loss=dice_loss,
    metrics=[iou_score],
)



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

history = model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=40,
                              validation_data=val_gen, validation_steps=val_steps,
                             verbose=1,
                             callbacks=callbacks_list)

test_gen = test_generator(batch_size=1)

model.load_weights('model.h5')
predictions = model.predict_generator(test_gen,
                                      steps=len(df_test),
                                      verbose=1)

preds_test_thresh = (predictions >= 0.7).astype(np.uint8)

preds_test_thresh.shape

print(preds_test_thresh.min())
print(preds_test_thresh.max())

mask = preds_test_thresh[3,:,:,0]
plt.imshow(mask, cmap='Reds', alpha=0.3)

true_mask = Y_test[3,:,:,0]
plt.imshow(true_mask, cmap='Blues', alpha=0.3)
true_mask = Y_test[3,:,:,0]
plt.imshow(true_mask, cmap='Blues', alpha=0.3)