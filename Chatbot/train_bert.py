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

VOCAB_SIZE = 28997

sess = tf.compat.v1.Session

print('Opening Conversations....')
# train = pd.read_csv(os.curdir + '/data/conversations.csv', dtype=str, sep='\*\*\|systemvaz\|\*\*', encoding='utf-8', comment=None, engine='python')

# txt_placeholder = tf.placeholder(dtype='string')

# Custom text pre-processing layer for inputs into BERT
class TextPrePro(tf.keras.layers.Layer):
    def create_tokenizer(self, vocab_file, do_lower_case=False):
        return tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    
    def __init__(self, **kwargs):
        self.trainable=False
        super(TextPrePro, self).__init__(**kwargs)

    def build(self, input_shape):
        self.tokenizer = self.create_tokenizer(os.curdir + '/data/vocab.txt', do_lower_case=False)
        super(TextPrePro, self).build(input_shape)

    def convert_sentence_to_features(self, sentence, tokenizer, max_seq_len):
        tokens = ['[CLS]']
        tokens.extend(tokenizer.tokenize(sentence))
        if len(tokens) > max_seq_len-1:
            tokens = tokens[:max_seq_len-1]
        tokens.append('[SEP]')
        
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        #Zero Mask till seq_length
        zero_mask = [0] * (max_seq_len-len(tokens))
        input_ids.extend(zero_mask)
        input_mask.extend(zero_mask)
        segment_ids.extend(zero_mask)
        
        return input_ids, input_mask, segment_ids

    def convert_sentences_to_features(self, sentences, tokenizer, max_seq_len=20):

        all_input_ids = []
        all_input_mask = []
        all_segment_ids = []
        
        for sentence in sentences:
            input_ids, input_mask, segment_ids = self.convert_sentence_to_features(sentence, tokenizer, max_seq_len)
            all_input_ids.append(input_ids)
            all_input_mask.append(input_mask)
            all_segment_ids.append(segment_ids)
    
        return all_input_ids, all_input_mask, all_segment_ids

    def call(self, x):
        txt = tf.compat.v1.Session().run(x)
        input_ids_vals, input_mask_vals, segment_ids_vals = self.convert_sentences_to_features(txt, self.tokenizer, 240)
        return input_ids_vals, input_mask_vals, segment_ids_vals


# Define Model
def BuildChatNetwork():
    encoder_inputs = tf.keras.layers.Input(shape=(1, ), dtype='string')
    enc_ids, enc_masks, enc_segments = TextPrePro() (encoder_inputs)

    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/1", trainable=False)
    _, sequence_output = bert_layer([enc_ids, enc_masks, enc_segments])
    
    _, state_h, state_c = tf.keras.layers.LSTM(200, return_state=True) (sequence_output)
    encoder_states = [state_h, state_c]

    decoder_inputs = tf.keras.layers.Input(shape=(1,  ), dtype='string')
    dec_ids, dec_masks, dec_segments = TextPrePro() (decoder_inputs)

    _, sequence_output2 = bert_layer([dec_ids, dec_masks, dec_segments])

    decoder_lstm = tf.keras.layers.LSTM(200, return_state=True, return_sequences=True)
    decoder_outputs, _ , _ = decoder_lstm(sequence_output2, initial_state=encoder_states)

    decoder_dense = tf.keras.layers.Dense(VOCAB_SIZE, activation=tf.keras.activations.softmax) 
    output = decoder_dense (decoder_outputs)

    model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output )
    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='sparse_categorical_crossentropy')

    return model

model = BuildChatNetwork()

print(model.summary())