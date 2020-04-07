import tensorflow as tf
import numpy as np
import pandas as pd
import tokenization
import random
import json
import h5py
import io
import os

max_features = 30000
max_length_q = 30
max_length_a = 30

hf5 = h5py.File(os.curdir + '/data/training_data_1milMsg_30kVocab_BERT.h5', 'w')
token_file = os.curdir + '/data/tokeniser_1mMsg_30kVocab_BERT.json'

def create_tokenizer(vocab_file, do_lower_case=False):
    return tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    

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

print('Opening Conversations....')
train = pd.read_csv(os.curdir + '/data/conversations.csv', dtype=str, sep='\*\*\|systemvaz\|\*\*', encoding='utf-8', comment=None, engine='python')
print('Printing Pandas dataframe...')
print(train)

print("Splitting to x and y (first 1mil)....")
x = train.iloc[0:1000000, 0].astype(str)
y = train.iloc[0:1000000, 1].astype(str)
# x = train.iloc[0:1000, 0].astype(str)
# y = train.iloc[0:1000, 1].astype(str)

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

tokenizer = create_tokenizer(os.curdir + '/data/vocab.txt', do_lower_case=False)

# Create Encoder inputs
print('Creating tokenised x (Encoder Inputs)....')
input_ids_vals, input_mask_vals, segment_ids_vals = convert_sentences_to_features(x, tokenizer, max_length_q)
print("Creating Training HF5 data...")
hf5.create_dataset('input_ids_vals_ENCin', data=input_ids_vals)
hf5.create_dataset('input_mask_vals_ENCin', data=input_mask_vals)
hf5.create_dataset('segment_ids_vals_ENCin', data=segment_ids_vals)
del x

# Tokenise text
print('Building Tokeniser....')
tokeniser = tf.keras.preprocessing.text.Tokenizer(num_words=max_features)
tokeniser.fit_on_texts(list(y))
VOCAB_SIZE = len(tokeniser.word_index) + 1
print( 'VOCAB SIZE : {}'.format(VOCAB_SIZE))
# Save Tokeniser
print('Saving tokeniser json to disk....')
tokeniser_json = tokeniser.to_json()
with io.open(token_file, 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokeniser_json, ensure_ascii=False))

# Create Decoder inputs
print('Creating tokenised y (Decoder Inputs)....')
y = tokeniser.texts_to_sequences(y)
y = tf.keras.preprocessing.sequence.pad_sequences(y, maxlen=maxlen_a, padding='post')
hf5.create_dataset('dec_in_data', data=y)
print("TOKENISED y....")
print(y)
print(y.shape)
print("Creating Training HF5 data...")

# Create Decoder outputs
print('Creating tokenised z (Decoder Outputs)....')
z = tokeniser.texts_to_sequences(y)
del y
z = tf.keras.preprocessing.sequence.pad_sequences(z, maxlen=max_length_a+1, padding='post')
z = np.delete(z, 0, 1)
hf5.create_dataset('dec_out_data', data=z)
print("TOKENISED z....")
print(y)
print(y.shape)
print('Saving Decoder Out HF5...')

# Save Tokeniser
token_file = os.curdir + '/data/tokeniser_1milMsg_30kVocab.json'
print('Saving tokeniser json to disk....')
tokeniser_json = tokeniser.to_json()
with io.open(token_file, 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokeniser_json, ensure_ascii=False))

hf5.close()
print("All Done!!!")