import pickle
from hw_models import comb_model1
import os
import numpy as np

entries = list()
for i in range(0,256):
    for j in range(0,256):
        res = comb_model1(i, j)
        entries.append([np.array([i, j]), np.array(res)])

pickle_dir = './pickles/'
pickle_file = 'comb1.pickle'

if not os.path.isdir(pickle_dir):
    os.makedirs(pickle_dir)

with open(os.path.join(pickle_dir,pickle_file), 'wb') as f:
    pickle.dump(entries, f)
