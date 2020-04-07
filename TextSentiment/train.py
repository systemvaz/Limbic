from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import random
import json
import os
import io

os.environ["CUDA_VISIBLE_DEVICES"]="1"

seed = 0
epochs = 20
batch_size = 512
max_length = 400
max_features = 20000

random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

train = np.loadtxt(os.curdir + '/data/twitter_train.csv', dtype=str, delimiter='*|*')

# Convert to lowercase
train = np.char.lower(train)
# Shuffle data
np.random.shuffle(train)
# Get split our text and labels
x = train[:, 1]
y = tf.keras.utils.to_categorical(train[:, 0], 2)

# Tokenise text
tokeniser = tf.keras.preprocessing.text.Tokenizer(num_words=max_features)
tokeniser.fit_on_texts(list(x))
x = tokeniser.texts_to_sequences(x)
x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=max_length)

# Save Tokeniser
tokeniser_json = tokeniser.to_json()
with io.open('tokeniser.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokeniser_json, ensure_ascii=False))

# Split into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=seed)

# Build our model!!
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(max_features, 150, input_length=max_length))

model.add(tf.keras.layers.SpatialDropout1D(0.2))

model.add(tf.keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

model.add(tf.keras.layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(2, activation='sigmoid'))

# Prepare model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'TxtSentiment_model.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                monitor='val_acc',
                                                verbose=1,
                                                save_best_only=True)

callbacks = [checkpoint]

opt = tf.keras.optimizers.Adam

# Compile and train
model.compile(loss='categorical_crossentropy', optimizer=opt(learning_rate=0.0001), metrics=['accuracy'])

model.fit(x_train, 
          y_train, 
          validation_data=(x_val, y_val), 
          epochs=epochs,
          batch_size=batch_size,
          callbacks=callbacks)