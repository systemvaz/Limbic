import numpy as np
import pandas as pd
import random
import json
import h5py
import io
import os

# print('Creating new H5 file....')
# hf5 = h5py.File(os.curdir + '/data/training_data.h5', 'w')

# print('Opening Conversations....')
# train = np.load(os.curdir + '/data/npy/dec_in_data.npy')

# print('Saving to HF5....')
# hf5.create_dataset('dec_in_data', data=train)

# print('Opening Conversations....')
# train = np.load(os.curdir + '/data/npy/enc_in_data.npy')

# print('Saving to HF5....')
# hf5.create_dataset('enc_in_data', data=train)

print('Opening H5 file....')
hf5 = h5py.File(os.curdir + '/data/training_data.h5')

print('Opening Conversations....')
train = np.load(os.curdir + '/data/npy/dec_out_data.npy')

print('Saving to HF5....')
hf5.create_dataset('dec_out_data', data=train)

hf5.close()