from arima.arima_pollution import arima, check_seasonality
from helpers.analyzer import plot_one_file
from pathlib import Path

from helpers.preparator import cut_csv

path = '%s/pollution_data/new_data' % Path(__file__).parent
path_to_file = '%s/Centar_O3.csv' % path

start_date = '2017-01-01 00:00:00'
end_date = '2018-01-02 00:00:00'

# plot_all_data('../pollution_data/new_data')
# plot_one_file(path_to_file, start=start_date, end=end_date)
# cut_csv(path_to_file, start=start_date, end=end_date)

arima(path_to_file, start=start_date, end=end_date)
# check_seasonality(path_to_file, start_date, end_date)
