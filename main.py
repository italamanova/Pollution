from analysis.analyzer import analyze

from analysis.trend_seasonality_checker import box_plot, check_seasonal_decomposition
from arima.arima_pollution import check_seasonality, analyze_data, my_auto_arima, pure_arima
from es.des import manual_es
from es.es import exponential_smoothing
from helpers.plotter import plot_one_file, plot_all_data
from pathlib import Path

from helpers.preparator import cut_csv, fill_nan, cut_last, fill_nan_rolling_mean, generate_features, remove_duplicates, \
    interpolate_nan, delete_outliers
from helpers.saver import get_autosave_path

# path = '%s/pollution_data/cut_data' % Path(__file__).parent
# path_to_file = '%s/Centar_PM25_4W.csv' % path
from helpers.visualizer import simple_plot
from lstm.lstm import my_lstm
from lstm.lstm_rolling_window import exp_lstm

path = '%s/pollution_data/experiment_data' % Path(__file__).parent
path_to_file = '%s/Centar_PM25_2012-01-01__2018-03-01_no_duplicates_no_outliers_inter_3D.csv' % path

# path = '%s/pollution_data/cut_data' % Path(__file__).parent
# path_to_file = '%s/Centar_PM25_4W_fill_mean.csv' % path


start_date = '2012-01-01 00:00:00'
end_date = '2018-03-01 00:00:00'

out_folder = 'experiment_data'
out_file = get_autosave_path(path_to_file, out_folder, '3D')

# ANALYSIS
# remove_duplicates(path_to_file, out_file)
# delete_outliers(path_to_file, out_file)
# interpolate_nan(path_to_file, out_file, start=start_date, end=end_date)
# cut_last(path_to_file, out_file, '2W')
analyze(path_to_file)


# METHODS

# ES
# exponential_smoothing(path_to_file)
# manual_es(path_to_file)

# ARIMA
# my_auto_arima(path_to_file)
# pure_arima(path_to_file)

# LSTM
# my_lstm(path_to_file)
# lstm_path_to_file = '%s/pollution_data/cut_data/Centar_PM25_fill_mean_year.csv' % Path(__file__).parent
# exp_lstm(lstm_path_to_file)

# ADDITIONAL
plot_one_file(path_to_file)

# HELPERS
# cut_csv(path_to_file, out_file, start=start_date, end=end_date)
# plot_all_data(path)
# fill_nan(path_to_file, out_file 'ffill')
# fill_nan_rolling_mean(path_to_file, out_file, 12, start=start_date, end=end_date)
# check_seasonality(path_to_file, start_date, end_date)
# box_plot(path_to_file)
