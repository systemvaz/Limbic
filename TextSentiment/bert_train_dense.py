from sklearn.model_selection import train_test_split
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import random
import h5py
import os

os.environ['TFHUB_CACHE_DIR'] = 'D:\Development\TensorFlow\TFHUB_CACHE'
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# tf.enable_eager_execution()

batch_size = 100
epochs = 100
seed = 20

random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

# Load Twitter BERT data
data = h5py.File(os.curdir + '/data/training_data_BERT.h5', 'r')

print('Loading train data...')
input_ids_TRAIN = np.array(data['input_ids_vals_TRAIN'])
mask_ids_TRAIN = np.array(data['input_mask_vals_TRAIN'])
segment_ids_TRAIN = np.array(data['segment_ids_vals_TRAIN'])
y_TRAIN = np.array(data['labels_TRAIN'])
print('Loading test data...')
input_ids_TEST = np.array(data['input_ids_vals_TEST'])
mask_ids_TEST = np.array(data['input_mask_vals_TEST'])
segment_ids_TEST = np.array(data['segment_ids_vals_TEST'])
y_TEST = np.array(data['labels_TEST'])


# Define model
input1 = tf.keras.layers.Input(shape=(240,), batch_size=batch_size, dtype="int32", name='input_data')
input2 = tf.keras.layers.Input(shape=(240,), batch_size=batch_size, dtype="int32", name='mask_data')
input3 = tf.keras.layers.Input(shape=(240,), batch_size=batch_size, dtype="int32", name='segment_data')

bert_layer = hub.KerasLayer("https://tfhub.dev/google/albert_xxlarge/2", trainable=True)
pooled_output, sequence_output = bert_layer([input1, input2, input3])
bert_output = sequence_output

flat1 = tf.keras.layers.Flatten()(bert_output)
dense1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)(flat1)
dpout1 = tf.keras.layers.Dropout(0.5)(dense1)

dense2 = tf.keras.layers.Dense(256, activation=tf.nn.relu)(dpout1)
dpout2 = tf.keras.layers.Dropout(0.5)(dense2)
output = tf.keras.layers.Dense(2, activation='sigmoid')(dpout2)

model = tf.keras.Model(inputs=[input1, input2, input3], outputs=output)

# Prepare model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'TxtSentiment_model_BERT.{epoch:03d}.h5'
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
model.compile(loss='categorical_crossentropy', optimizer=opt(learning_rate=0.001), metrics=['accuracy'])

print(model.summary())

model.fit([input_ids_TRAIN, mask_ids_TRAIN, segment_ids_TRAIN], y_TRAIN, 
          validation_data=([input_ids_TEST, mask_ids_TEST, segment_ids_TEST], y_TEST), 
          epochs=epochs,
          batch_size=batch_size,
          callbacks=callbacks)