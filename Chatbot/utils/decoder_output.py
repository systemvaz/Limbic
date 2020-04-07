from keras_preprocessing.text import tokenizer_from_json
import tensorflow as tf
import numpy as np
import pandas as pd
import random
import json
import h5py
import io
import os

token_path = os.curdir + '/data/tokeniser.json'

print('Opening Conversations....')
train = np.load(os.curdir + '/data/dec_in_data.npy')

print("ORIGINAL y....")
print(train)
print(train.shape)
print("")

print('Loading Tokeniser....')
with open(token_path) as f:
    data = json.load(f)
    tokeniser = tokenizer_from_json(data)

VOCAB_SIZE = len(tokeniser.word_index) + 1
print( 'VOCAB SIZE : {}'.format(VOCAB_SIZE))

del tokeniser

maxlen_answers = max( [ len(x) for x in train ] )
print('Maxliength: {}'.format(maxlen_answers))

print('Removing <START> token....')
train = np.delete(train, 0, 1)

print("<START> Removed....")
print(train)
print(train.shape)
print("")

print('Padding....')
train = tf.keras.preprocessing.sequence.pad_sequences(train , maxlen=maxlen_answers , padding='post' )
print("Padding added....")
print(train)
print(train.shape)
print("")

# print('Converting to one-hot....')
# train = tf.keras.utils.to_categorical(train , VOCAB_SIZE )
# print('Creating final Numpy array...')
# train = np.array(train)

print("TOKENISED z....")
print(train)
print(train.shape)
print("")

print('Saving final output....')
hf5 = h5py.File(os.curdir + '/data/dec_out_data_preproc.h5', 'w')
hf5.create_dataset('dec_out_data', data=train)
hf5.close()

print('DONE!!')