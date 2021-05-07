import numpy as np
import os

data = np.genfromtxt(os.curdir + '/data/twitter_all.csv', delimiter='",', usecols=(0,5), dtype=str)
data = np.char.strip(data, chars='"')
data[data == '4'] = '1'
np.savetxt(os.curdir + "/data/twitter_train.csv", data, fmt='%s', delimiter='*|*')