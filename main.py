from helpers.analyzer import plot_one_file
from pathlib import Path

path = Path(__file__).parent
path_to_file = '%s/pollution_data/new_data' % path

# plot_all_data('../pollution_data/new_data')
plot_one_file('%s/Centar_CO.csv' % path_to_file, start='2013-03-01', end='2013-03-02')
