from helpers.analyzer import plot_one_file
from pathlib import Path

from helpers.preparator import cut_csv

path = '%s/pollution_data/new_data' % Path(__file__).parent
path_to_file = '%s/Centar_CO.csv' % path

start_date = '2013-03-01 00:00:00'
end_date = '2013-03-02 00:00:00'

# plot_all_data('../pollution_data/new_data')
plot_one_file(path_to_file, start=start_date, end=end_date)
cut_csv(path_to_file, start=start_date, end=end_date)

