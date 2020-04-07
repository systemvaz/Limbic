from sklearn.model_selection import train_test_split
import tensorflow_hub as hub
import tensorflow as tf
import tokenization
import random
import numpy as np
import pandas as pd
import h5py
import json
import sys
import os
import io

VOCAB_SIZE = 30000
max_length_q = 30
max_length_a = 30
batch_size = 100

os.environ['TFHUB_CACHE_DIR'] = 'D:\Development\TensorFlow\TFHUB_CACHE'
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Load data
data = h5py.File(os.curdir + '/data/training_data_1milMsg_30kVocab_BERT.h5', 'r')

print('Loading Encoder data...')
input_ids_enc = np.array(data['input_ids_vals_ENCin'])
mask_ids_enc = np.array(data['input_mask_vals_ENCin'])
segment_ids_enc = np.array(data['segment_ids_vals_ENCin'])

print(input_ids_enc.shape)
print(mask_ids_enc.shape)
print(segment_ids_enc.shape)

print('Loading Decoder data...')
input_ids_dec = np.array(data['input_ids_vals_DECin'])
mask_ids_dec = np.array(data['input_mask_vals_DECin'])
segment_ids_dec = np.array(data['segment_ids_vals_DECin'])

print(input_ids_dec.shape)
print(mask_ids_dec.shape)
print(segment_ids_dec.shape)

print('Loading Decoder output data')
output_dec = np.array(data['input_ids_vals_DECout'])

print(output_dec)
print(output_dec.shape)

# Define Model
def BuildChatNetwork():
    input1_enc = tf.keras.layers.Input(shape=(max_length_q,), batch_size=batch_size, dtype="int32", name='enc_input_data')
    input2_enc = tf.keras.layers.Input(shape=(max_length_q,), batch_size=batch_size, dtype="int32", name='enc_mask_data')
    input3_enc = tf.keras.layers.Input(shape=(max_length_q,), batch_size=batch_size, dtype="int32", name='enc_segment_data')

    albert_inputs_enc = dict(input_ids=input1_enc, input_mask=input2_enc, segment_ids=input3_enc)
    bert_layer1 = hub.KerasLayer("https://tfhub.dev/google/albert_base/2", trainable=True, signature='tokens', output_key='sequence_output')
    
    sequence_output1 = bert_layer1(albert_inputs_enc)
    _, state_h, state_c = tf.keras.layers.LSTM(768, return_state=True) (sequence_output1)
    encoder_states = [state_h, state_c]

    input1_dec = tf.keras.layers.Input(shape=(max_length_a,), batch_size=batch_size, dtype="int32", name='dec_input_data')
    input2_dec = tf.keras.layers.Input(shape=(max_length_a,), batch_size=batch_size, dtype="int32", name='dec_mask_data')
    input3_dec = tf.keras.layers.Input(shape=(max_length_a,), batch_size=batch_size, dtype="int32", name='dec_segment_data')

    albert_inputs_dec = dict(input_ids=input1_dec, input_mask=input2_dec, segment_ids=input3_dec)
    bert_layer2 = hub.KerasLayer("https://tfhub.dev/google/albert_base/2", trainable=True, signature='tokens', output_key='sequence_output')

    sequence_output2 = bert_layer2(albert_inputs_dec)
    decoder_lstm = tf.keras.layers.LSTM(768, return_state=True, return_sequences=True)
    decoder_outputs, _ , _ = decoder_lstm(sequence_output2, initial_state=encoder_states)

    decoder_dense = tf.keras.layers.Dense(VOCAB_SIZE, activation=tf.keras.activations.softmax) 
    output = decoder_dense (decoder_outputs)

    model = tf.keras.models.Model([input1_enc, input2_enc, input3_enc,
                                   input1_dec, input2_dec, input3_dec], output)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='sparse_categorical_crossentropy')

    return model

model = BuildChatNetwork()
print(model.summary())

# Prepare model saving directory.
save_dir = os.path.join(os.getcwd(), 'models')
model_name = 'ChatBot_model_ALBERT_1mil.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                monitor='loss',
                                                verbose=1,
                                                save_best_only=True)

callbacks = [checkpoint]

#Train
model.fit([input_ids_enc, mask_ids_enc, segment_ids_enc, input_ids_dec, mask_ids_dec, segment_ids_dec], output_dec, 
          batch_size=batch_size, 
          epochs=30, 
          callbacks=callbacks) 
