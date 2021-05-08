import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
import random
import numpy as np
import h5py
import json
import sys
import os
import io

import tokenization

seed = 20
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


def create_tokenizer(vocab_file, do_lower_case=True):
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


def main():
    print('Loading data...')
    train = np.loadtxt(os.curdir + '/data/twitter_train.csv', dtype=str, delimiter='*|*')
    tokenizer = create_tokenizer(os.curdir + '/data/vocab.txt', do_lower_case=False)

    # Process dataset
    print('Processing dataset for BERT...')
    x = train[:, 1]
    y = tf.keras.utils.to_categorical(train[:, 0], 2)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=seed, shuffle=True)
    sentences = list(x_train)
    input_ids_vals, input_mask_vals, segment_ids_vals = convert_sentences_to_features(sentences, tokenizer, 240)

    print("Creating Training HF5 data...")
    hf5 = h5py.File(os.curdir + '/data/training_data_BERT.h5', 'w')
    hf5.create_dataset('input_ids_vals_TRAIN', data=input_ids_vals)
    hf5.create_dataset('input_mask_vals_TRAIN', data=input_mask_vals)
    hf5.create_dataset('segment_ids_vals_TRAIN', data=segment_ids_vals)
    hf5.create_dataset('labels_TRAIN', data=y_train)

    print('input_ids_vals TRAIN shape: {}'.format(hf5['input_ids_vals_TRAIN'].shape))
    print('input_mask_vals TRAIN shape: {}'.format(hf5['input_mask_vals_TRAIN'].shape))
    print('segment_ids_vals TRAIN shape: {}'.format(hf5['segment_ids_vals_TRAIN'].shape))
    print('labels TRAIN shape: {}'.format(hf5['labels_TRAIN'].shape))

    sentences = list(x_val)
    input_ids_vals, input_mask_vals, segment_ids_vals = convert_sentences_to_features(sentences, tokenizer, 240)

    print("Creating Testing HF5 data...")
    hf5.create_dataset('input_ids_vals_TEST', data=input_ids_vals)
    hf5.create_dataset('input_mask_vals_TEST', data=input_mask_vals)
    hf5.create_dataset('segment_ids_vals_TEST', data=segment_ids_vals)
    hf5.create_dataset('labels_TEST', data=y_val)

    print('input_ids_vals TEST shape: {}'.format(hf5['input_ids_vals_TEST'].shape))
    print('input_mask_vals TEST shape: {}'.format(hf5['input_mask_vals_TEST'].shape))
    print('segment_ids_vals TEST shape: {}'.format(hf5['segment_ids_vals_TEST'].shape))
    print('labels TEST shape: {}'.format(hf5['labels_TEST'].shape))

    print('DONE!!')


if __name__ == "__main__":
    main()