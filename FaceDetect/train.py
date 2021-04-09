# -----------------------------------------------
# Author: Alex Varano
# Train new Resnet56 model on the FER2013 dataset
# -----------------------------------------------

from __future__ import absolute_import, division, print_function, unicode_literals

from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import keras
import glob
import os
# model.py....
from model import resnet

# Set for multi-gpu training
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

# Model and data variables/constants here
def get_model_consts():
    BATCH_SIZE = 256
    EPOCHS = 100
    NUM_CLASSES = 7
    n = 18
    version = 2
    depth = n * 9 + 2
    model_type = 'ResNet%dv%d' % (depth, version)

    return BATCH_SIZE, EPOCHS, NUM_CLASSES, depth, model_type

def get_data_consts():
    data_file = os.curdir + '/dataset/fer2013.csv'
    image_dir = os.curdir + '/dataset/fer_images/'

    return data_file, image_dir

# Create numpy array of image data
def get_image_array(image_dir):
    filelist = sorted(glob.glob(image_dir + '*.png'), key=lambda name: int(name[len(image_dir):-4]))
    image_array = np.array([np.array(Image.open(fname)) for fname in filelist])
    image_array = image_array.reshape(image_array.shape[0],48,48,1)
    input_shape = image_array.shape[1:]
    print('Input shape: ', input_shape)
    print('Total shape: ', image_array.shape)

    return input_shape, image_array

# Create numpy array of label data
def get_label_array(data_file):
    label_data = pd.read_csv(data_file, usecols=['emotion'])
    label_array = label_data.to_numpy()

    classes_dict = {'Angry':0, 'Disgust':1,'Fear':2, 'Happy':3, 
                    'Sad':4, 'Surprise':5, 'Neutral':6}

    return classes_dict, label_array

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

#Print some shape info on our data
def print_shapes(x_train, y_train, x_val, y_val):
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print('x_val shape: ', x_val.shape)
    print('y_val shape: ', y_val.shape)

#Loss rate scheduler function. Loss rate determined by epoch number
def lr_schedule(EPOCH):
    lr = 1e-2
    if EPOCH > 100:
        lr *= 0.5e-3
    elif EPOCH > 90:
        lr *= 1e-3
    elif EPOCH > 25:
        lr *= 1e-2
    elif EPOCH > 12:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

#Image augmentation object for better model generalisation
def image_augmentation(x_train):
    datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            zca_epsilon=1e-06,
            rotation_range=22.5,
            zoom_range = 0,
            width_shift_range=0.0,
            height_shift_range=0.0,
            shear_range=0.,
            channel_shift_range=0.,
            fill_mode='nearest',
            cval=0.,
            horizontal_flip=True,
            vertical_flip=False)

    datagen.fit(x_train)
    return datagen

# Build model!
def compile_model(input_shape, depth, NUM_CLASSES):
    model = resnet(input_shape=input_shape, depth=depth, num_classes=NUM_CLASSES)
    opt = tf.keras.optimizers.Adam

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt(lr=lr_schedule(0)),
                  metrics=['accuracy'])
    
    model.summary()
    print('ResNet%dv2' % (depth))

    return model

# Prepare model saving directory.
def model_save_dir(model_type):
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'EmoDetect_%s_model.{epoch:03d}.h5' % model_type

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    filepath = os.path.join(save_dir, model_name)
    return filepath

# Prepare callbacks for model save checkpoints and learning rate adjustment.
def model_callbacks(filepath):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                    monitor='val_accuracy',
                                                    verbose=1,
                                                    save_best_only=True)

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                      factor=np.sqrt(0.1),
                                                      cooldown=0,
                                                      patience=5,
                                                      min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler]
    return callbacks

#Visualise ou4 training results
def training_results(history, EPOCHS):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
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

def main():
    # Get model and data variables
    BATCH_SIZE, EPOCHS, NUM_CLASSES, depth, model_type = get_model_consts()
    data_file, image_dir = get_data_consts()

    # Get image and label array
    input_shape, image_array = get_image_array(image_dir)
    _, label_array = get_label_array(data_file)

    # Normalise image data and display sample images
    image_array = image_array.astype('float32') / 255
    image_sample(image_array[:25], label_array[:25])

    # Prepare training data split and display shapes
    x_train, x_val,\
    y_train, y_val = train_test_split(image_array, label_array, test_size=0.20, random_state=2)
    print_shapes(x_train, y_train, x_val, y_val)

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val = keras.utils.to_categorical(y_val, NUM_CLASSES)

    #Generate and display sample of augmentated images
    datagen = image_augmentation(x_train)
    imgaug, lblaug = next(datagen.flow(x_train, y_train, batch_size=25))
    image_sample(imgaug, lblaug)

    # Define optimiser and compile our ResNet model
    model = compile_model(input_shape, depth, NUM_CLASSES)

    # Prepare model save directory and callbacks
    filepath = model_save_dir(model_type)
    callbacks = model_callbacks(filepath)

    #Train our model!
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                                  validation_data=(x_val, y_val),
                                  epochs=EPOCHS,
                                  callbacks=callbacks)

    # Graph training accuracy/loss
    training_results(history, EPOCHS)


if __name__ == "__main__":
    main()
