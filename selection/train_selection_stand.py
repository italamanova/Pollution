from pathlib import Path

from helpers.preparator import get_data, get_data_with_box_cox
from helpers.visualizer import simple_plot
from selection.train_selection import select_train

METHOD_NAME = 'Exponential Smoothing'

path_prepared = '%s/pollution_data/centar' % Path(__file__).parents[1]
path_to_file_prepared = '%s/Centar_PM25_prepared.csv' % path_prepared
df, lambda_ = get_data_with_box_cox(path_to_file_prepared)
train = 24*365
test = train + 24*10+48
df = df.iloc[train:test]
# series = df[df.columns[0]].values
# get_train_test(series)
select_train(df, 24*10, 1, 24, lambda_, method_name='arima', out_file_name='2')
