from pathlib import Path

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from helpers.preparator import get_data


def get_train_test(series):
    tscv = TimeSeriesSplit(n_splits=20)
    for train_index, test_index in tscv.split(series):
        print("TRAIN:", train_index, "TEST:", test_index)
        train, test = df.iloc[train_index], df.iloc[test_index]
        # print(train, test)


path_prepared = '%s/pollution_data/centar' % Path(__file__).parents[1]
path_to_file_prepared = '%s/Centar_O3_prepared.csv' % path_prepared
df = get_data(path_to_file_prepared).iloc[:100]
series = df[df.columns[0]].values
get_train_test(series)
