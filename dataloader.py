import os
import h5py
import numpy as np
from collections import namedtuple
import tqdm
import torchfile
import pickle
import random
import matplotlib.pyplot as plt
import torch.utils.data as data
from PIL import Image
from os import path
from warnings import warn

import torchvision.transforms as transforms
import torchvision.transforms.functional as F


Pickle = 'comb1.pickle'

labels_histo_pickle = 'labels_histo.pickle'

pickle_dir = './pickles'

class Dataset(data.Dataset):
    def __init__(self, train=True, pickle_name=None, transforms=None):
        self.train = train
        self.transforms = transforms
        pickle_filename = Pickle
        pickle_path = os.path.join(pickle_dir, pickle_filename)
        with open(pickle_path, 'rb') as f:
            self.dataset = pickle.load(f)
        if train:
            self.dataset = self.dataset[len(self.dataset) // 10:]
        else:
            self.dataset = self.dataset[:len(self.dataset) // 10]


    def __getitem__(self, index):
        x, y = self.dataset[index]

        return x, y


    def __len__(self):
        return len(self.dataset)

    def name(self):
        return 'Dataset'



def example():
    train_dir = '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Sintel/left_filtered_adapted_all_but_alley_1'
    test_dir = '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Sintel/left_filtered_adapted_alley_1'
    label_dir = '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Sintel/depth_flatten'
    train_files = os.listdir(train_dir)
    train_filelist = [os.path.join(train_dir, img) for img in train_files]
    label_filelist = [img.replace(train_dir, label_dir).replace('.png','.dpt') for img in train_filelist]

    dataset = Dataset(train_filelist, label_filelist, train=True)
    print('Dataset:', dataset)


if __name__ == '__main__':
    example()
