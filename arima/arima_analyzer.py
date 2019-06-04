import glob
import os

from visualizer import plot
import statsmodels.api as sm
import plotly.plotly as py
from plotly.graph_objs import *
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as FF

import pandas as pd

plotly.tools.set_credentials_file(username='talamash', api_key='NVqgaGN3OpMYcqXncLOw')


def plot_all_data(folder_path):
    os.chdir(folder_path)

    extension = 'csv'
    all_filenames = [i for i in glob.glob('Centar_*.{}'.format(extension))]

    for file in all_filenames:
        dataset = pd.read_csv(file, index_col=0)
        data = dataset

        data.index = pd.to_datetime(data.index)

        plot(data, ylabel=data.columns[0], title=file)
        print('%s description' % file, data[data.columns[0]].describe())
        print(len(data))


def plot_one_file(file, start='2008-01-01', end='2018-03-09'):
    dataset = pd.read_csv(file, index_col=0)
    dataset.index = pd.to_datetime(dataset.index)
    data = dataset[start:end]

    plot(data, ylabel=data.columns[0], title=file)
    print('%s description' % file, data[data.columns[0]].describe())
    print(len(data))


# plot_all_data('../pollution_data/new_data')
plot_one_file('../pollution_data/new_data/Centar_CO.csv', start='2013-03-01', end='2013-03-02')
# plot_to_plotly('../pollution_data/new_data')
