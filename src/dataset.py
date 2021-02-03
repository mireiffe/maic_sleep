'''
data loader {img, label}
'''
import logging
from os.path import join

import pandas as pd
from PIL import Image
import pickle
from torch.utils.data import Dataset


class SleepMAIC(Dataset):
    '''
    make a list of data
    '''
    def __init__(self, dir_data, split, transform):
        
        # self.dir_data = dir_data
        # self.split = split
        # self.transform = transform

        # # load data information from .csv file
        # info_train = pd.read_csv(join(self.dir_data, "trainset-for_user.csv"),
        #     names=['cases', 'datapoints', 'labels'])
        # css = info_train.loc[:, 'cases']
        # sz_css = css.value_counts()

        # init_idx = sz_css.iloc[:split[0]].sum()
        # end_idx = sz_css.iloc[:split[1]].sum()
        # self.use_files = info_train[init_idx:end_idx]

        preset = {range(0, 700): 'train_0700', range(700, 796): 'train_700796'}

        with open(join(self.dir_data, preset[split]), 'rb') as f:
            self.use_file = pickle.load(f)

        logging.info(f'Creating dataset with {len(self.use_files)} examples')

    def __len__(self):
        return len(self.use_files)

    def __getitem__(self, index):
        # load file
        case, datapoint, label = self.use_files.iloc[index]
        img = Image.open(join(self.dir_data, *[case, datapoint])).convert('L')

        # transformimg data
        input = self.transform(img)
        
        return input, label
