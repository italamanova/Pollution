import glob
import os

import pandas as pd
import plotly


from helpers.visualizer import plot, simple_plot

plotly.tools.set_credentials_file(username='talamash', api_key='NVqgaGN3OpMYcqXncLOw')


def plot_all_data(folder_path):
    os.chdir(folder_path)

    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

    for file in all_filenames:
        print(file)
        dataset = pd.read_csv(file, index_col=0)
        data = dataset

        data.index = pd.to_datetime(data.index)
        simple_plot(data, ylabel=data.columns[0], title=file)


def plot_one_file(file, start='2008-01-01', end='2018-03-09'):
    dataset = pd.read_csv(file, index_col=0)
    dataset.index = pd.to_datetime(dataset.index)
    data = dataset[start:end]

    plot(data, ylabel=data.columns[0], title=file)




