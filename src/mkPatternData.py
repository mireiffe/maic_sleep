from os.path import join
import pandas as pd

dir_data = 'data/Sleep'

info_train = pd.read_csv(
    join(dir_data, f"{type}set-for_user.csv"),
    names=['class', 'datapoints', 'labels']
)

css:pd.Series = info_train.loc[:, 'cases']
sz_css = css.value_counts()

preset = {'WAKE': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM':4}

for i in range(len(sz_css)):
    init_i = sz_css.iloc[:i].sum()
    term_i = sz_css.iloc[:i+1].sum()
    
    patt = info_train.iloc[init_i:term_i, 2]

    print(patt)
    break