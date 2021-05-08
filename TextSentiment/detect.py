import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from keras_preprocessing.text import tokenizer_from_json

import tokenization

model_path = os.curdir + '/saved_models/TxtSentiment_model_BERT.001.h5'
vocab_file = os.curdir + '/data/vocab.txt'


def create_tokenizer(vocab_file, do_lower_case=True):
    return tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    

def convert_sentence_to_features(sentence, tokeniser, max_seq_len):
    input_ids_arr = []
    input_mask_arr = []
    segment_ids_arr = []
    
    tokens = ['[CLS]']
    tokens.extend(tokeniser.tokenize(sentence))
    if len(tokens) > max_seq_len-1:
        tokens = tokens[:max_seq_len-1]
    tokens.append('[SEP]')
    
    segment_ids = [0] * len(tokens)
    input_ids = tokeniser.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    #Zero Mask till seq_length
    zero_mask = [0] * (max_seq_len-len(tokens))
    input_ids.extend(zero_mask)
    input_mask.extend(zero_mask)
    segment_ids.extend(zero_mask)

    input_ids_arr.append(input_ids)
    input_mask_arr.append(input_mask)
    segment_ids_arr.append(segment_ids)
    
    return np.array(input_ids_arr), np.array(input_mask_arr), np.array(segment_ids_arr)


def main():
    sentiment_dict = {0:'Negative', 1:'Positive'}
    model = tf.keras.models.load_model((model_path), custom_objects={'KerasLayer':hub.KerasLayer}, compile=True)
    # model.compile()

    tokeniser = create_tokenizer(vocab_file)

    while True:
        # sentence = np.array([input("Enter Sentence: ")], dtype=str)
        sentence = input("Enter sentence: ")

        if sentence == 'q':
            break

        sentence_to_bert = convert_sentence_to_features(sentence, tokeniser, 240)
        sentiment = model.predict(sentence_to_bert)
        print("Sentence is: {}".format(sentiment_dict.get(np.argmax(sentiment))))


if __name__ == "__main__":
    main()