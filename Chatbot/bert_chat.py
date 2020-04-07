from keras_preprocessing.text import tokenizer_from_json
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import tokenization
import random
import json
import io
import os

VOCAB_SIZE = 3115
max_length_a = 30
max_length_q = 30
batch_size = 1


input1_enc = tf.keras.layers.Input(shape=(None,), dtype="int32", name='enc_input_data')
input2_enc = tf.keras.layers.Input(shape=(None,), dtype="int32", name='enc_mask_data')
input3_enc = tf.keras.layers.Input(shape=(None,), dtype="int32", name='enc_segment_data')

albert_inputs_enc = dict(input_ids=input1_enc, input_mask=input2_enc, segment_ids=input3_enc)
bert_layer1 = hub.KerasLayer("https://tfhub.dev/google/albert_base/2", trainable=True, signature='tokens', output_key='sequence_output')

sequence_output1 = bert_layer1(albert_inputs_enc)
_, state_h, state_c = tf.keras.layers.LSTM(200, return_state=True) (sequence_output1)
encoder_states = [state_h, state_c]

input1_dec = tf.keras.layers.Input(shape=(None,), dtype="int32", name='dec_input_data')
input2_dec = tf.keras.layers.Input(shape=(None,), dtype="int32", name='dec_mask_data')
input3_dec = tf.keras.layers.Input(shape=(None,), dtype="int32", name='dec_segment_data')

albert_inputs_dec = dict(input_ids=input1_dec, input_mask=input2_dec, segment_ids=input3_dec)
bert_layer2 = hub.KerasLayer("https://tfhub.dev/google/albert_base/2", trainable=True, signature='tokens', output_key='sequence_output')

sequence_output2 = bert_layer2(albert_inputs_dec)
decoder_lstm = tf.keras.layers.LSTM(200, return_state=True, return_sequences=True)
decoder_outputs, _ , _ = decoder_lstm(sequence_output2, initial_state=encoder_states)

decoder_dense = tf.keras.layers.Dense(VOCAB_SIZE, activation=tf.keras.activations.softmax) 
output = decoder_dense (decoder_outputs)

model = tf.keras.models.Model([input1_enc, input2_enc, input3_enc,
                                input1_dec, input2_dec, input3_dec], output)

model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='sparse_categorical_crossentropy')


# Create Inference Models
def make_inference_models():
    print('Creating inference models....')  
    encoder_model = tf.keras.models.Model([input1_enc, input2_enc, input3_enc], encoder_states)
    
    decoder_state_input_h = tf.keras.layers.Input(shape=(768 ,))
    decoder_state_input_c = tf.keras.layers.Input(shape=(768 ,))
    
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder_outputs, state_h, state_c = decoder_lstm(sequence_output2 , initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.models.Model(
        ([input1_dec, input2_dec, input3_dec] + decoder_states_inputs),
        [decoder_outputs] + decoder_states)

    return encoder_model, decoder_model


def create_tokenizer(vocab_file, do_lower_case=False):
    return tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


# Helper function
def convert_sentence_to_features(sentence, tokenizer, max_seq_len):
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

def convert_sentences_to_features(sentences, tokenizer, max_seq_len=20):
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    
    for sentence in sentences:
        input_ids, input_mask, segment_ids = convert_sentence_to_features(sentence, tokenizer, max_seq_len)
        all_input_ids.append(input_ids)
        all_input_mask.append(input_mask)
        all_segment_ids.append(segment_ids)
    
    return all_input_ids, all_input_mask, all_segment_ids


model.load_weights(os.curdir + '/models/ChatBot_model_ALBERT_1mil_TEST.018.h5')
# enc_model, dec_model = make_inference_models()

tokenizer = create_tokenizer(os.curdir + '/data/vocab.txt', do_lower_case=False)
token_path = os.curdir + '/data/tokeniser_1milMsg_20kVocab.json'

with open(token_path) as f:
    data = json.load(f)
    tokeniser = tokenizer_from_json(data)


# Begin chat!
for _ in range(10):
    question = input('Enter question: ')
    inid, inmask, inseg = convert_sentences_to_features(question, tokenizer, max_length_q)
    inid = np.asarray(inid)
    inmask = np.asarray(inmask)
    inseg = np.asarray(inseg)

    empty_id, empty_mask, empty_seg = convert_sentences_to_features('start', tokenizer, 3)
    empty_id = np.asarray(empty_id)
    empty_mask = np.asarray(empty_mask)
    empty_seg = np.asarray(empty_seg)
    # empty_target_seq[0, 0] = tokeniser.word_index['start']
    stop_condition = False
    decoded_translation = ''
    while not stop_condition :
        dec_outputs = model.predict([inid, inmask, inseg, empty_id, empty_mask, empty_seg])
        print(dec_outputs)
        print(dec_outputs.shape)
        sampled_word_index = np.argmax(dec_outputs[0, -1, :])
        print(sampled_word_index)
        sampled_word = None
        for word, index in tokeniser.word_index.items() :
            if sampled_word_index == index :
                decoded_translation += ' {}'.format(word)
                print(decoded_translation)
                sampled_word = word
        
        if sampled_word == 'end' or len(decoded_translation.split()) > max_length_a:
            stop_condition = True
            
        # empty_id, empty_mask, empty_seg = convert_sentences_to_features(sampled_word_index, tokenizer, 3)
        # empty_id = np.asarray(empty_id)
        # empty_mask = np.asarray(empty_mask)
        # empty_seg = np.asarray(empty_seg)

    # print(decoded_translation)