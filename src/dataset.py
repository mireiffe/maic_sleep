'''
data loader {img, label}
'''
import logging
import os
import pickle
from glob import glob
from time import time

from tqdm import tqdm
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class SleepMAIC(Dataset):
    '''
    make a list of data
    '''
    preset = {'train': (0, 700), 'valid': (701, 800), 'test': (0, 200)}

    def __init__(self, dir_data, type_data, transform, transform_train=None, transform_target=None):
        self.dir_data = dir_data
        self.transform = transform
        self.transform_train = transform_train
        self.transform_target = transform_target

        # total file list
        total_files =[os.path.splitext(file) for file in os.listdir(dir_data)]

        # files with <type_data>
        _tp = self.preset[type_data]
        self.files = [
            tfs for tfs in total_files
            if int(tfs[0][-4:]) >= _tp[0]
            and int(tfs[0][-4:]) < _tp[1]
        ]
        logging.info(f'Creating dataset with {len(self.files)} examples')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        name_file = self.files[index][0]
        ext_file = self.files[index][1]
        path_file = os.path.join(self.dir_data, name_file + ext_file)

        # load file
        img = Image.open(path_file).convert('L')
        #===============================
        # To be implemented
        label = 0
        #===============================

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if isinstance(img, np.ndarray):
            img = 255 * img / img.max()
            img = Image.fromarray(img.astype('uint8'), mode='L')

        # transformimg data
        input = self.transform(img)
        
        return input, label
