from pathlib import Path

import pandas as pd

from helpers.visualizer import plot

parent_dir_path = Path(__file__).parents[1]

def arima(file, start='2008-01-01', end='2018-03-09'):
    dataset = pd.read_csv(file, index_col=0)
    dataset.index = pd.to_datetime(dataset.index)

    data = dataset.loc['1985-01-01':'2018-01-01']
    print(data.head())

    data.columns = ['Energy Production']

    plot(data, ylabel='Energy consumption', title='Energy consumption')
    print('PM10 description', data['Energy Production'].describe())
    print(len(data))
