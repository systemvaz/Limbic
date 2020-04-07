import tensorflow as tf
import h5py
import os

VOCAB_SIZE = 20000

print('Loading hf5 file...')
hf5 = h5py.File(os.curdir + '/data/training_data.h5', 'r')
data = hf5['dec_out_data']

sliced = data[0:10000, :]

print(sliced)
print(sliced.shape)

# print('Performing Onehot encoding...')
# onehot_answers = tf.keras.utils.to_categorical(decoder_output_data , VOCAB_SIZE )

# print('Creating new hf5 dataset...')
# hf5.create_dataset('dec_out_onehot_data', data=onehot_answers)

# print('Done. Closing...')
hf5.close()