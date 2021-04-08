# ----------------------------------------------
# Author: Alex Varano
# Convert fer2013.csv pixel data into PNG images
# ----------------------------------------------

import pandas as pd
import numpy as np
import cv2
import os

data_file = os.curdir + '/dataset/fer2013.csv'
fer_data = pd.read_csv(data_file, delimiter=',')

for index,row in fer_data.iterrows():
    pixels = np.asarray(list(row['pixels'].split(' ')),dtype=np.uint8)
    img = pixels.reshape((48, 48))
    pathname = os.path.join('fer_images', str(index)+'.png')
    cv2.imwrite(pathname,img)
    print('image saved ias {}'.format(pathname))