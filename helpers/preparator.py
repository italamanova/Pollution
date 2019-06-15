import os
import glob
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.stats import stats

from helpers.saver import df_to_csv
from helpers.visualizer import simple_plot

parent_dir_path = Path(__file__).parents[1]


def get_data(file):
    df = pd.read_csv(file)
    df = df.set_index('time')
    df.index = pd.to_datetime(df.index)
    return df


def split_csv(folder_path):
    os.chdir(folder_path)

    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

    for file in all_filenames:
        df = pd.read_csv(file, index_col=0)
        filename = file[:-4]
        for column in df.columns:
            if column != 'time':
                new_df = df.loc[:, ['time', column]].copy()
                new_df = new_df.rename(columns={column: filename})
                new_df.to_csv('./new_data/%s_%s' % (column, file), index=False, encoding='utf-8-sig')


def cut_csv(file, out_file_name, start, end):
    df = pd.read_csv(file, index_col=0)
    new_df = df.loc[start:end]
    new_df.to_csv(out_file_name, encoding='utf-8-sig')


def fill_nan(file, out_file_name, method, start=None, end=None):
    df = pd.read_csv(file, index_col=0)
    if start and end:
        df = df.loc[start:end]
    df.fillna(method=method, inplace=True)
    filled_data = df.dropna(how='any', inplace=False)
    filled_data.to_csv(out_file_name, encoding='utf-8-sig')


def fill_nan_rolling_mean(file, out_file_name, window, start=None, end=None):
    df = pd.read_csv(file, index_col=0)
    simple_plot(df, title='Initial dataset')
    col_name = df.columns[0]
    if start and end:
        df = df.loc[start:end]
    df['rollmean'] = df[col_name].rolling(window, center=True, min_periods=1).mean()

    df['update'] = df['rollmean']
    df['update'].update(df[col_name])
    filled_data = df.dropna(how='any', inplace=False)
    simple_plot(filled_data, title='Rolling mean')
    filled_data.to_csv(out_file_name, columns=[filled_data.columns[0]], index=True, encoding='utf-8-sig')


def cut_last(file, out_file_name, last_parameter):
    df = pd.read_csv(file, index_col=0)
    df.index = pd.to_datetime(df.index)

    new_df = df.last(last_parameter)
    new_df.to_csv(out_file_name, encoding='utf-8-sig')


def generate_features(file, out_file_name):
    df = pd.read_csv(file, index_col=0)
    df.index = pd.to_datetime(df.index)

    df['month'] = [df.index[i].month for i in range(len(df))]
    df['year'] = [df.index[i].year for i in range(len(df))]
    df.to_csv(out_file_name, encoding='utf-8-sig')


def remove_duplicates(file, out_file):
    df = get_data(file)
    new_df = df.loc[~df.index.duplicated(keep='first')]
    df_to_csv(new_df, out_file)
    return new_df


def delete_outliers(df, m=3):
    new_df = df[(np.abs(stats.zscore(df)) < m).all(axis=1)]
    return new_df
