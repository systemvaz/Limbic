import numpy as np
import os

# data = pd.read_csv(, usecols=[0, 5])

data = np.genfromtxt(os.curdir + '/data/twitter_all.csv', delimiter='",', usecols=(0,5), dtype=str)
np.savetxt(os.curdir + "/data/twitter_train.csv", data, fmt='%s', delimiter='*|*')

