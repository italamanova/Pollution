from analysis.analyzer import analyze_average

from analysis.trend_seasonality_checker import box_plot
from arima.arima_pollution import check_seasonality, analyze_data, my_auto_arima, pure_arima
from es.es import exponential_smoothing
from helpers.plotter import plot_one_file, plot_all_data
from pathlib import Path

from helpers.preparator import cut_csv, fill_nan, cut_last, fill_nan_rolling_mean, generate_features
from helpers.saver import get_autosave_path

# path = '%s/pollution_data/cut_data' % Path(__file__).parent
# path_to_file = '%s/Centar_PM25_4W.csv' % path
from helpers.visualizer import simple_plot
from lstm.lstm import my_lstm
from lstm.lstm_rolling_window import exp_lstm

# path = '%s/pollution_data/cut_data' % Path(__file__).parent
# path_to_file = '%s/Centar_PM25_4W_fill_mean.csv' % path


path = '%s/pollution_data/cut_data' % Path(__file__).parent
path_to_file = '%s/Rektorat_CO_2015-01-01__2018-03-01.csv' % path

# path = '%s/data' % Path(__file__).parent
# path_to_file = '%s/pas.csv' % path

start_date = '2015-01-01 00:00:00'
end_date = '2018-03-01 00:00:00'

out_file = get_autosave_path(path_to_file, '2015-01-01__2018-03-01')
# generate_features(path_to_file, out_file)
# cut_csv(path_to_file, out_file, start=start_date, end=end_date)


# plot_all_data(path)
# plot_one_file(path_to_file)
# fill_nan(path_to_file, out_file 'ffill')
# cut_last(path_to_file, out_file, '6M')
# fill_nan_rolling_mean(path_to_file, out_file, 12, start=start_date, end=end_date)

# analyze_data(path_to_file)
# my_auto_arima(path_to_file)
# pure_arima(path_to_file, start=start_date, end=end_date)
# check_seasonality(path_to_file, start_date, end_date)

# my_lstm(path_to_file)
# lstm_path_to_file = '%s/pollution_data/cut_data/Centar_PM25_fill_mean_year.csv' % Path(__file__).parent
# exp_lstm(lstm_path_to_file)

# box_plot(path_to_file)
analyze_average(path_to_file)

# check_decomposition(path_to_file)

# exponential_smoothing(path_to_file)
