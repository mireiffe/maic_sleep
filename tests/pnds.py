import pandas as pd
import numpy as np

info_train = pd.read_csv("data/Sleep/trainset-for_user.csv", names=['cases', 'datapoints', 'labels'])

css = info_train.loc[:, 'cases']
dpts = info_train.loc[:, 'datapoints']
lbls = info_train.loc[:, 'labels']

key_css = css.unique()
dict_css = {int(cs[-4:]): cs for cs in key_css}
order_css = np.sort(dict.keys())

print(dict_css)
