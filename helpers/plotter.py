import glob
import os

import chart_studio
import pandas
import pandas as pd
import plotly
from plotly import graph_objs as go

from matplotlib import pyplot

from helpers.visualizer import plot_to_file, simple_plot

chart_studio.tools.set_credentials_file(username='italamanova', api_key='aCzH4J9rwXrrjrvAPNGO')


def plot_all_data(folder_path):
    os.chdir(folder_path)

    extension = 'csv'
    all_filenames = [i for i in glob.glob('Rektorat*.{}'.format(extension))]

    for file in all_filenames:
        dataset = pd.read_csv(file, index_col=0)
        data = dataset

        data.index = pd.to_datetime(data.index)
        simple_plot(data, ylabel=data.columns[0], title=file)


def plot_one_file(file, start=None, end=None, title=''):
    filename_w_ext = os.path.basename(file)
    filename, file_extension = os.path.splitext(filename_w_ext)

    dataset = pd.read_csv(file, index_col=0)
    dataset.index = pd.to_datetime(dataset.index)
    if start and end:
        data = dataset[start:end]
    else:
        data = dataset
    simple_plot(data, ylabel=data.columns[0], title=title)


def plot_average(data, col_name):
    weekly = data.resample('M').sum()
    plot_to_file(weekly, ylabel='%s_weekly' % col_name, title='Air pollution')


def plot_rolling_average(data, col_name):
    rolling = data.rolling(window=5040)
    rolling_mean = rolling.mean()
    data.plot()
    rolling_mean.plot(color='red')
    pyplot.show()


def plot_boxplot(data):
    data.boxplot()
    pyplot.show()


def plot_heatmap(x, y, z):
    data = [go.Heatmap(x=x, y=y, z=z, colorscale='Viridis')]
    chart_studio.plotly.plot(data, filename='pandas-heatmap')
