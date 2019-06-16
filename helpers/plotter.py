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


def plot_one_file(file, start=None, end=None):
    filename_w_ext = os.path.basename(file)
    filename, file_extension = os.path.splitext(filename_w_ext)

    dataset = pd.read_csv(file, index_col=0)
    dataset.index = pd.to_datetime(dataset.index)
    if start and end:
        data = dataset[start:end]
    else:
        data = dataset
    simple_plot(data, ylabel=data.columns[0], title='')
