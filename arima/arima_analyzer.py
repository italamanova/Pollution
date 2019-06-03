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


def plot_to_plotly(folder_path):
    os.chdir(folder_path)

    extension = 'csv'
    all_filenames = [i for i in glob.glob('Centar_CO.{}'.format(extension))]

    for file in all_filenames:
        df = pd.read_csv(file, index_col=0)
        print(df)
        df.index = pd.to_datetime(df.index)
        df = df.loc['2010-01-01':'2010-12-01']

        trace = go.Scatter(x=[df.index], y=[df.columns[0]],
                           name=file)
        layout = go.Layout(title=file,
                           plot_bgcolor='rgb(230, 230,230)',
                           showlegend=True)
        fig = go.Figure(data=[trace], layout=layout)

        py.plot(fig, filename=file)


# plot_all_data('../pollution_data/new_data')
plot_to_plotly('../pollution_data/new_data')
