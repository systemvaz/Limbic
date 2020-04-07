from tensorflow.python.framework.ops import disable_eager_execution
import tensorflow_hub as hub
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
os.environ['TFHUB_CACHE_DIR'] = 'D:\Development\TensorFlow\TFHUB_CACHE'

VOCAB_SIZE = 30000
EPOCHS = 30
BATCH_SIZE = 50

print('Opening Conversations....')
train = pd.read_csv(os.curdir + '/data/conversations.csv', dtype=str, sep='\*\*\|systemvaz\|\*\*', encoding='utf-8', comment=None, engine='python')
print('Printing Pandas dataframe...')
print(train)

print("Splitting to x and y....")
encoder_input_data = train.iloc[0:1000000, 0].astype(str)
decoder_input_data = train.iloc[0:1000000, 1].astype(str)
del train
print('Creating z....')
print('Opening H5 file....')
hf5 = h5py.File(os.curdir + '/data/training_data_1mMsg_30kVocab.h5', 'r')
decoder_output_data = np.array(hf5['dec_out_data'])
hf5.close()

print(encoder_input_data)
print(encoder_input_data.shape)
print(decoder_input_data)
print(decoder_input_data.shape)
print(decoder_output_data)
print(decoder_output_data.shape)

# Creating TF data dataset
# print('Creating TF datasets....')
# encoder_input = tf.data.Dataset.from_tensor_slices(encoder_input_data)
# decoder_input = tf.data.Dataset.from_tensor_slices(decoder_input_data)
# decoder_output = tf.data.Dataset.from_tensor_slices(decoder_output_data)

# Loading TF Hub layer
print('Loading TF Hub module layer....')
embedding_layer = hub.KerasLayer("https://tfhub.dev/google/Wiki-words-500/2",
                                 input_shape=[], 
                                 dtype=tf.string,
                                 trainable=False) 

# Define Model
encoder_inputs = tf.keras.layers.Input((), dtype=tf.string)
encoder_embedding = embedding_layer (encoder_inputs)

encoder_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(500, return_state=True))
encoder_outputs, fw_state_h, fw_state_c, bw_state_h, bw_state_c =  encoder_lstm(encoder_embedding)
state_h = tf.keras.layers.Concatenate() ([fw_state_h, bw_state_h])
state_c = tf.keras.layers.Concatenate() ([fw_state_c, bw_state_c])
encoder_states = [state_h, state_c]

decoder_inputs = tf.keras.layers.Input((), dtype=tf.string)
decoder_embedding = embedding_layer (decoder_inputs)

decoder_lstm = tf.keras.layers.LSTM(1000, return_state=True, return_sequences=True)
decoder_outputs, _ , _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

decoder_dense = tf.keras.layers.Dense(VOCAB_SIZE, activation=tf.keras.activations.softmax) 
output = decoder_dense (decoder_outputs)

model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output)

model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='sparse_categorical_crossentropy')
model.summary()

# Prepare model saving directory.
save_dir = os.path.join(os.getcwd(), 'models')
model_name = 'ChatBot_model_bidirectional.{epoch:03d}.h5'
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

# # Create Inference Models
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