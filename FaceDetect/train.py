from __future__ import absolute_import, division, print_function, unicode_literals

from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from sklearn.model_selection import train_test_split
from PIL import Image

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import keras
import glob
import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"

BATCH_SIZE = 128
EPOCHS = 200
NUM_CLASSES = 7

n = 6
version = 2
depth = n * 9 + 2
model_type = 'ResNet%dv%d' % (depth, version)

data_file = os.curdir + '/dataset/fer2013.csv'
image_dir = os.curdir + '/fer_images'

# Create numpy array of images
filelist = sorted(glob.glob(image_dir + '/*.png'), key=lambda name: int(name[13:-4]))
image_array = np.array([np.array(Image.open(fname)) for fname in filelist])
image_array = image_array.reshape(image_array.shape[0],48,48,1)
input_shape = image_array.shape[1:]

print('Input shape: ', input_shape)
print('Total shape: ', image_array.shape)

# Create numpy array of labels
label_data = pd.read_csv(data_file, usecols=['emotion'])
label_array = label_data.to_numpy()

classes_dict = {'Angry':0, 'Disgust':1,'Fear':2, 'Happy':3, 
                'Sad':4, 'Surprise':5, 'Neutral':6}

# Normalise image data
image_array = image_array.astype('float32') / 255

#Display a sample of training images
def image_sample(sample, label):
    i = 0
    sample = sample.reshape(sample.shape[0],48,48)
    plt.figure(figsize=(32,32))
    for image in sample:
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap='gray')
        plt.xlabel(label[i])
        i += 1
    plt.show()

image_sample(image_array[:25], label_array[:25])

x_train, x_val, y_train, y_val = train_test_split(image_array, label_array, test_size=0.20, random_state=2)

#Print some info on our data
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_val shape: ', x_val.shape)
print('y_val shape: ', y_val.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_val = keras.utils.to_categorical(y_val, NUM_CLASSES)

#Loss rate scheduler function. Loss rate determined by epoch number
def lr_schedule(EPOCH):
    lr = 1e-3
    if EPOCH > 180:
        lr *= 0.5e-3
    elif EPOCH > 160:
        lr *= 1e-3
    elif EPOCH > 120:
        lr *= 1e-2
    elif EPOCH > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr
#End loss rate function

#ResNet layer
def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):

    conv = tf.keras.layers.Conv2D(num_filters,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding='same',
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=l2(1e-4))

    x = inputs

    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x)
        if activation is not None:
            x = tf.keras.layers.Activation(activation)(x)
    else:
        if batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x)
        if activation is not None:
            x = tf.keras.layers.Activation(activation)(x)
        x = conv(x)

    return x
#End ResNet layer

#Define our ResNet model
def resnet(input_shape, depth, num_classes=NUM_CLASSES):
    num_filters_in = 16
    num_res_blocks = int((depth -2) / 9)
    inputs = tf.keras.Input(shape=input_shape)
    x = resnet_layer(inputs=inputs, num_filters=num_filters_in, conv_first=True)

    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:
                    strides = 2

            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)

            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)

            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)

            if res_block == 0:
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)

            x = tf.keras.layers.add([x, y])

        num_filters_in = num_filters_out

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
        y = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        return model
#End of ResNet definition

#Image augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=0,
        zoom_range = 0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=True,
        vertical_flip=False)

datagen.fit(x_train)

#Display a sample of augmentated images
imgaug, lblaug = next(datagen.flow(x_train, y_train, batch_size=25))
image_sample(imgaug, lblaug)

#Define optimiser and compile our ResNet model
model = resnet(input_shape=input_shape, depth=depth)
opt = tf.keras.optimizers.Adam

model.compile(loss='categorical_crossentropy',
              optimizer=opt(lr=lr_schedule(0)),
              metrics=['accuracy'])

model.summary()
print('ResNet%dv2' % (depth))

# Prepare model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'EmoDetect_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                monitor='val_acc',
                                                verbose=1,
                                                save_best_only=True)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),
                                                  cooldown=0,
                                                  patience=5,
                                                  min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

#Train our model!
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                              validation_data=(x_val, y_val),
                              epochs=EPOCHS,
                              callbacks=callbacks)

#Visualise out training results
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('./training_results.png')
plt.show()