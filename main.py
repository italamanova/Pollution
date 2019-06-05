from arima.arima_pollution import arima, check_seasonality, analyze_data
from helpers.analyzer import plot_one_file, plot_all_data
from pathlib import Path

from helpers.preparator import cut_csv

path = '%s/pollution_data/new_data' % Path(__file__).parent
path_to_file = '%s/Centar_PM25.csv' % path

start_date = '2014-01-01 00:00:00'
end_date = '2018-03-01 00:00:00'

# plot_all_data(path)
# plot_one_file(path_to_file, start=start_date, end=end_date)
cut_csv(path_to_file, start=start_date, end=end_date)

# analyze_data(path_to_file, start=start_date, end=end_date)
# arima(path_to_file, start=start_date, end=end_date)
# check_seasonality(path_to_file, start_date, end_date)
