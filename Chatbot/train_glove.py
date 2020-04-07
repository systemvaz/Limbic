from tensorflow.python.framework.ops import disable_eager_execution
from keras_preprocessing.text import tokenizer_from_json
import tensorflow as tf
import numpy as np
import pandas as pd
import random
import json
import h5py
import io
import os

disable_eager_execution()

os.environ["CUDA_VISIBLE_DEVICES"]="1"

VOCAB_SIZE = 30000
EPOCHS = 30
BATCH_SIZE = 100

print('Opening H5 file....')
hf5 = h5py.File(os.curdir + '/data/training_data_5mMsg_30kVocab.h5', 'r')

print('Loading tokeniser...')
token_path = os.curdir + '/data/tokeniser_5mMsg_30kVocab.json'
with open(token_path) as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)


print('Obtaining datasets....')
encoder_input_data = hf5['enc_in_data']
decoder_input_data = hf5['dec_in_data']
decoder_output_data = hf5['dec_out_data']

encoder_input_data = encoder_input_data[:,:]
decoder_input_data = decoder_input_data[:,:]
decoder_output_data = decoder_output_data[:,:]

print(encoder_input_data)
print(encoder_input_data.shape)
print(decoder_input_data)
print(decoder_input_data.shape)
print(decoder_output_data)
print(decoder_output_data.shape)

print('Creating embeddings from Glove...')
embeddings_index = dict()
glove = open(os.curdir + '/data/glove.6B.300d.txt', encoding='utf8')
for line in glove:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
glove.close()

embedding_matrix = np.zeros((VOCAB_SIZE, 300))
for word, index in tokenizer.word_index.items():
    if index > VOCAB_SIZE - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector


# Define Model
encoder_inputs = tf.keras.layers.Input(shape=(None, ))
embedding_layer = tf.keras.layers.Embedding(VOCAB_SIZE, 300, mask_zero=True, weights=[embedding_matrix], trainable=False) 
encoder_embedding = embedding_layer (encoder_inputs)

encoder_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(300, return_state=True))
encoder_outputs, fw_state_h, fw_state_c, bw_state_h, bw_state_c =  encoder_lstm(encoder_embedding)
state_h = tf.keras.layers.Concatenate() ([fw_state_h, bw_state_h])
state_c = tf.keras.layers.Concatenate() ([fw_state_c, bw_state_c])
encoder_states = [state_h, state_c]

decoder_inputs = tf.keras.layers.Input(shape=(None,  ))
decoder_embedding = embedding_layer (decoder_inputs)

decoder_lstm = tf.keras.layers.LSTM(300*2, return_state=True, return_sequences=True)
decoder_outputs, _ , _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

decoder_dense = tf.keras.layers.Dense(VOCAB_SIZE, activation=tf.keras.activations.softmax) 
output = decoder_dense (decoder_outputs)

model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output)
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='sparse_categorical_crossentropy')

model.summary()

# Prepare model saving directory.
save_dir = os.path.join(os.getcwd(), 'models')
model_name = 'ChatBot_model_glove.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                monitor='loss',
                                                verbose=1,
                                                save_best_only=True)

callbacks = [checkpoint]

# Train model!
model.fit([encoder_input_data, decoder_input_data], 
           decoder_output_data, 
           batch_size=BATCH_SIZE, 
           epochs=EPOCHS, 
           callbacks=callbacks) 

# Create Inference Models
# def make_inference_models():
#     print('Creating inference models....')  
#     encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)
    
#     decoder_state_input_h = tf.keras.layers.Input(shape=(200 ,))
#     decoder_state_input_c = tf.keras.layers.Input(shape=(200 ,))
    
#     decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
#     decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding , initial_state=decoder_states_inputs)
#     decoder_states = [state_h, state_c]
#     decoder_outputs = decoder_dense(decoder_outputs)
#     decoder_model = tf.keras.models.Model(
#         [decoder_inputs] + decoder_states_inputs,
#         [decoder_outputs] + decoder_states)

#     encoder_model.save(os.curdir + '/models/EncDec/enc_model_epoch.h5')
#     decoder_model.save(os.curdir + '/models/EncDec/dec_model_epoch.h5')

# make_inference_models()

print("Finished!!")