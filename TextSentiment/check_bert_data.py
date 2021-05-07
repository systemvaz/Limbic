import h5py
import os
import numpy as np

data = h5py.File(os.curdir + '/data/training_data_BERT.h5', 'r')

input_ids = np.array(data['input_ids_vals_TEST'])
mask_ids = np.array(data['input_mask_vals_TEST'])
segment_ids = np.array(data['segment_ids_vals_TEST'])

print("Input Ids.......")
print(input_ids)
print("Shape: {}".format(input_ids.shape))
print("----------------------------------------")
print("Input Masks.......")
print(mask_ids)
print("Shape: {}".format(mask_ids.shape))
print("----------------------------------------")
print("Segment Ids........")
print(segment_ids)
print("Shape: {}".format(segment_ids.shape))
print("----------------------------------------")