from keras_preprocessing.text import tokenizer_from_json
import tensorflow as tf
import numpy as np
import random
import json
import io
import os

# maxlen_questions = 92
# maxlen_answers = 94
VOCAB_SIZE = 30000
maxlen_answers = 92
maxlen_questions = 90

token_path = os.curdir + '/data/tokeniser_5mMsg_30kVocab.json'
model_path = os.curdir + '/models/ChatBot_model_glove.001.h5'

print('Loading NN models...')
# model = tf.keras.models.load_model(os.curdir + '/models/ChatBot_model.001.h5', compile=True)

print('Loading tokeniser...')
with open(token_path) as f:
    data = json.load(f)
    tokeniser = tokenizer_from_json(data)

# Helper function
def str_to_tokens(sentence : str):
    words = sentence.lower().split()
    tokens_list = list()
    for word in words:
        tokens_list.append(tokeniser.word_index[word]) 

    return tf.keras.preprocessing.sequence.pad_sequences( [tokens_list] , maxlen=maxlen_questions , padding='post')

# Define Model
encoder_inputs = tf.keras.layers.Input(shape=(None, ))
embedding_layer = tf.keras.layers.Embedding(VOCAB_SIZE, 300, mask_zero=True, trainable=False) 
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

# Create Inference Models
def make_inference_models():
    print('Creating inference models....')  
    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)
    
    decoder_state_input_h = tf.keras.layers.Input(shape=(300*2,))
    decoder_state_input_c = tf.keras.layers.Input(shape=(300*2,))
    
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding , initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.models.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    return encoder_model, decoder_model


model.load_weights(model_path)
enc_model, dec_model = make_inference_models()

# Begin chat!
for _ in range(10):
    states_values = enc_model.predict(str_to_tokens(input('Enter question : ')))
    empty_target_seq = np.zeros((1, 1))
    empty_target_seq[0, 0] = tokeniser.word_index['start']
    stop_condition = False
    decoded_translation = ''
    while not stop_condition :
        dec_outputs, h, c = dec_model.predict([empty_target_seq] + states_values)
        # print(dec_outputs)
        # print(dec_outputs.shape)
        sampled_word_index = np.argmax(dec_outputs[0, -1, :])
        sampled_word = None
        for word, index in tokeniser.word_index.items() :
            if sampled_word_index == index :
                decoded_translation += ' {}'.format(word)
                sampled_word = word
        
        if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
            stop_condition = True
            
        empty_target_seq = np.zeros((1 , 1))  
        empty_target_seq[0, 0] = sampled_word_index
        states_values = [h, c ] 

    print(decoded_translation)