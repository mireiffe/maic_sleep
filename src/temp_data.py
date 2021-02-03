'''
data loader {img, label}
'''
import logging
from os.path import join

import pickle
import pandas as pd


def load_save():
    dir_data = "data/Sleep"

    # load data information from .csv file
    info_train = pd.read_csv(join(dir_data, "trainset-for_user.csv"),
        names=['cases', 'datapoints', 'labels'])
    css = info_train.loc[:, 'cases']
    sz_css = css.value_counts()

    split = [0, 700]

    init_idx = sz_css.iloc[:split[0]].sum()
    end_idx = sz_css.iloc[:split[1]].sum()
    use_files = info_train[init_idx:end_idx]

    logging.info(f'Creating dataset with {len(use_files)} examples')

    for i in range(len(use_files)):
        case, dtp, lbl = use_files.iloc[i, :]
        data = {i: (case, dtp, lbl)}

    with open("data/train_0700", 'wb') as f:
        pickle.dump(data, f)