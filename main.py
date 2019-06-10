from arima.arima_pollution import check_seasonality, analyze_data, my_auto_arima, pure_arima
from helpers.analyzer import plot_one_file, plot_all_data
from pathlib import Path

from helpers.preparator import cut_csv, fill_nan, cut_last, fill_nan_rolling_mean
from helpers.saver import get_autosave_path

# path = '%s/pollution_data/cut_data' % Path(__file__).parent
# path_to_file = '%s/Centar_PM25_4W.csv' % path
from helpers.visualizer import simple_plot
from lstm.lstm import my_lstm

path = '%s/pollution_data/cut_data' % Path(__file__).parent
path_to_file = '%s/Centar_PM25_4W_fill_mean.csv' % path

# path = '%s/data' % Path(__file__).parent
# path_to_file = '%s/pas.csv' % path

start_date = '2014-01-01 00:00:00'
end_date = '2018-03-01 00:00:00'

out_file = get_autosave_path(path_to_file, 'fill_mean')

# plot_all_data(path)
# plot_one_file(path_to_file)
# cut_csv(path_to_file, start=start_date, end=end_date)
# fill_nan(path_to_file, out_file 'ffill')
# cut_last(path_to_file, out_file, '6M')
# fill_nan_rolling_mean(path_to_file, out_file, 12)

# analyze_data(path_to_file)
# my_auto_arima(path_to_file)
# pure_arima(path_to_file, start=start_date, end=end_date)
# check_seasonality(path_to_file, start_date, end_date)

my_lstm(path_to_file)