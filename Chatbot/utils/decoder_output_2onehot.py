from keras_preprocessing.text import tokenizer_from_json
import tensorflow as tf
import numpy as np
import pandas as pd
import random
import json
import h5py
import io
import os

VOCAB_SIZE = 2567566
print('Loading data....')
hf5 = h5py.File(os.curdir + '/data/dec_out_data_preproc.h5', 'r')
train = hf5['dec_out_data']

print('Converting to one-hot....')
train = tf.keras.utils.to_categorical(train , VOCAB_SIZE)
print('Creating final Numpy array...')
train = np.array(train)
print("Array created....")
print(train)
print(train.shape)
print('Saving Numpy file....')
np.save(os.curdir + '/data/npy/dec_out_data.npy', train)