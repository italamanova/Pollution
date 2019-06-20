import datetime
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

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
                new_df.to_csv('./every_station_data/%s_%s' % (column, file), index=False, encoding='utf-8-sig')


def cut_csv(file, out_file, start, end):
    df = pd.read_csv(file, index_col=0)
    new_df = df.loc[start:end]
    new_df.to_csv(out_file, encoding='utf-8-sig')


def cut_last(df, last_parameter):
    df.index = pd.to_datetime(df.index)
    new_df = df.last(last_parameter)
    return new_df


def cut_dataframe(df, start, end):
    return df.loc[start:end].copy()


def cut_dataframe_by_period(df, start, hours):
    end = start + datetime.timedelta(hours=hours)
    return df.loc[start:end].copy()


def fill_nan(df, method, start=None, end=None):
    if start and end:
        df = df.loc[start:end]
    df.fillna(method=method, inplace=True)
    filled_data = df.dropna(how='any', inplace=False)
    return filled_data


def fill_nan_rolling_mean(df, window, start=None, end=None):
    simple_plot(df, title='Initial dataset')
    col_name = df.columns[0]
    if start and end:
        df = df.loc[start:end]
    df['rollmean'] = df[col_name].rolling(window, center=True, min_periods=1).mean()

    df['update'] = df['rollmean']
    df['update'].update(df[col_name])
    filled_data = df.dropna(how='any', inplace=False)
    simple_plot(filled_data, title='Rolling mean')
    return filled_data


def interpolate_nan(df):
    interpolated_data = df.interpolate(method='linear')
    return interpolated_data


def generate_features(df):
    df.index = pd.to_datetime(df.index)
    df['month'] = [df.index[i].month for i in range(len(df))]
    df['year'] = [df.index[i].year for i in range(len(df))]
    return df


def remove_duplicates(df):
    new_df = df.loc[~df.index.duplicated(keep='first')]
    return new_df


def replace(group, stds):
    group[np.abs(group - group.mean()) > stds * group.std()] = np.nan
    return group


def delete_outliers(df, m=2):
    mask = (df - df.mean()).abs() > m * df.std()
    new_df = df.mask(mask)
    return new_df


def sdd_missing_dates(df):
    df_date_range = pd.date_range(start=df.index[0], end=df.index[-1], freq='H')
    new_df = df.reindex(df_date_range).rename_axis('time')
    return new_df


def differencing(df):
    pass
