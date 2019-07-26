from pathlib import Path

from helpers.preparator import get_data, get_data_with_box_cox
from helpers.saver import print_to_file
from helpers.visualizer import simple_plot
from main_experiment.selection import run_select

path_prepared = '%s/data/centar' % Path(__file__).parents[1]
path_to_file_prepared = '%s/Centar_PM25_prepared.csv' % path_prepared

path_out = '%s/data/main_experiment' % Path(__file__).parents[1]
m_out_file = '%s/%s.json' % (path_out, 'arima_experiment')
df, lambda_ = get_data_with_box_cox(path_to_file_prepared)

train_window = 24 * 10
step = 1
test_window = 24
rolling_window = test_window

train_start_index = 24 * 365
test_index = train_start_index + train_window + test_window + rolling_window
df = df.iloc[train_start_index:test_index]

METHOD_NAME = 'arima'
# METHOD_NAME = 'es'

print_to_file(m_out_file, [])
run_select(df, train_window, step, test_window, rolling_window, lambda_, m_out_file, method_name=METHOD_NAME)
