import tensorflow as tf
import numpy as np
import pandas as pd
import random
import json
import h5py
import io
import os

max_features = 30000

hf5 = h5py.File(os.curdir + '/data/training_data_1mMsg_30kVocab.h5', 'w')
token_file = os.curdir + '/data/tokeniser_1mMsg_30kVocab.json'

print('Opening Conversations....')
train = pd.read_csv(os.curdir + '/data/conversations.csv', dtype=str, sep='\*\*\|systemvaz\|\*\*', encoding='utf-8', comment=None, engine='python')
print('Printing Pandas dataframe...')
print(train)

print("Splitting to x and y (first 1mil)....")
x = train.iloc[0:500000, 0].astype(str)
y = train.iloc[0:500000, 1].astype(str)
# x = train.iloc[:, 0].astype(str)
# y = train.iloc[:, 1].astype(str)
del train
print('Converting to lowercase....')
x.str.lower()
y.str.lower()

print("ORIGINAL x....")
print(x)
print(x.shape)
print("ORIGINAL y....")
print(y)
print(y.shape)
print("")

print("Converting to lists....")
x.values.tolist()
y.values.tolist()

# Tokenise text
print('Building Tokeniser....')
tokeniser = tf.keras.preprocessing.text.Tokenizer()
tokeniser.fit_on_texts(list(y))
VOCAB_SIZE = len(tokeniser.word_index) + 1
print( 'VOCAB SIZE : {}'.format(VOCAB_SIZE))
# Save Tokeniser
print('Saving tokeniser json to disk....')
tokeniser_json = tokeniser.to_json()
with io.open(token_file, 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokeniser_json, ensure_ascii=False))

# Create Encoder inputs
print('Creating tokenised x (Encoder Inputs)....')
x = tokeniser.texts_to_sequences(x)
maxlen_questions = max([len(i) for i in x])
x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=maxlen_questions, padding='post')
hf5.create_dataset('enc_in_data', data=x)
print("TOKENISED x....")
print(x)
print(x.shape)
del x

# Create Decoder inputs
print('Creating tokenised y (Decoder Inputs)....')
y = tokeniser.texts_to_sequences(y)
maxlen_answers = max([len(i) for i in y])
y = tf.keras.preprocessing.sequence.pad_sequences(y, maxlen=maxlen_answers, padding='post')
hf5.create_dataset('dec_in_data', data=y)
print("TOKENISED y....")
print(y)
print(y.shape)

# Create Decoder outputs
print('Creating tokenised z (Decoder Outputs)....')
z = np.delete(y, 0, 1)
del y
print('Printing z before processing....')
print(z)
print(z.shape)
z = tf.keras.preprocessing.sequence.pad_sequences(z, maxlen=maxlen_answers, padding='post')
hf5.create_dataset('dec_out_data', data=z)
print("TOKENISED z....")
print(z)
print(z.shape)
print("")

hf5.close()
print("All Done!!!")