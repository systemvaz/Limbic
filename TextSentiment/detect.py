import os
import json
import numpy as np
import tensorflow as tf
from keras_preprocessing.text import tokenizer_from_json

max_length = 400
max_features = 20000

model_path = os.curdir + '/models/TxtSentiment_model.008.h5'
token_path = os.curdir + '/models/tokeniser.json'

sentiment_dict = {0:'Negative', 1:'Positive'}
model = tf.keras.models.load_model(model_path, compile=True)

with open(token_path) as f:
    data = json.load(f)
    tokeniser = tokenizer_from_json(data)

while True:
    sentence = np.array([input("Enter Sentence: ")], dtype=str)

    if sentence == 'q':
        break

    sentence = tokeniser.texts_to_sequences(sentence)
    sentence = tf.keras.preprocessing.sequence.pad_sequences(sentence, maxlen=max_length)
    sentiment = model.predict_classes(sentence)
    print('Sentence is: ', sentiment_dict.get(int(sentiment), "Don't know!!"))