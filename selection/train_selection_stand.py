from pathlib import Path

from helpers.preparator import get_data
from selection.train_selection import select_train

METHOD_NAME = 'Exponential Smoothing'

path_prepared = '%s/pollution_data/centar' % Path(__file__).parents[1]
path_to_file_prepared = '%s/Centar_PM25_prepared.csv' % path_prepared
df = get_data(path_to_file_prepared).iloc[:50]
# series = df[df.columns[0]].values
# get_train_test(series)
select_train(df, 30, 1, 10)
