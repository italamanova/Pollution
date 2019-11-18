from analysis.analyzer import analyze
from analysis.batch_analysis import analyze_batch

from analysis.trend_seasonality_checker import box_plot, check_seasonal_decomposition, analyze_rolling
from es.des import manual_es
from es.es import exponential_smoothing_from_file
from helpers.converter import str_to_datetime
from helpers.dataset_preparator import prepare_csv
from helpers.plotter import plot_one_file, plot_all_data
from pathlib import Path

from helpers.preparator import cut_csv, fill_nan, cut_last, fill_nan_rolling_mean, generate_features, remove_duplicates, \
    interpolate_nan, delete_outliers, cut_csv_by_period
from helpers.saver import get_autosave_path

# path = '%s/data/cut_data' % Path(__file__).parent
# path_to_file = '%s/Centar_PM25_prepared.csv' % path
from helpers.visualizer import simple_plot
from lstm.lstm import my_lstm
from lstm.lstm_rolling_window import exp_lstm



# path = '%s/data/every_station_data' % Path(__file__).parent
# path_to_file = '%s/Centar_PM10.csv' % path
from myarima.arima import my_auto_arima

path = '%s/old/data' % Path(__file__).parent
path_to_file = '%s/pollution2.csv' % path

path_prepared = '%s/data/centar' % Path(__file__).parent
path_to_file_prepared = '%s/Centar_PM25_prepared.csv' % path_prepared

start = '2016-01-01 00:00:00'
end = '2017-12-21 23:00:00'

start_datetime = str_to_datetime(start)
end_datetime = str_to_datetime(end)

# out_folder = 'candidates_checking'
out_folder = 'cut_data'
out_file = get_autosave_path(path_to_file_prepared, out_folder, '2016_2017')

# prepare_csv(path_to_file, out_file, start=None, end=None)
# cut_csv(path_to_file_prepared, out_file, start=start, end=end)
# cut_csv_by_period(path_to_file_prepared, out_file, start=start_datetime, period=365*24)
analyze(path_to_file_prepared)

# METHODS

# print('\n ES')
# exponential_smoothing_from_file(path_to_file_prepared)
# print('\n ARIMA')
my_auto_arima(path_to_file_prepared, 24)
# print('\n LSTM')
# my_lstm(path_to_file)

# ES
# manual_es(path_to_file)

# ARIMA
# pure_arima(path_to_file_prepared)

# LSTM
# lstm_path_to_file = '%s/data/cut_data/Centar_PM25_fill_mean_year.csv' % Path(__file__).parent
# exp_lstm(lstm_path_to_file)

# ADDITIONAL
# plot_one_file(path_to_file_prepared)

# plot_one_file(path_to_file_prepared)

# HELPERS
# cut_csv(path_to_file, out_file, start=start_date, end=end_date)
# plot_all_data(path)
# fill_nan(path_to_file, out_file 'ffill')
# fill_nan_rolling_mean(path_to_file, out_file, 12, start=start_date, end=end_date)
# check_seasonality(path_to_file, start_date, end_date)
# box_plot(path_to_file)

# analyze_batch(path_prepared)



