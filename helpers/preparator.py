import os
import glob
import pandas as pd
from pathlib import Path

from analysis.analyzer import get_data
from helpers.visualizer import simple_plot

parent_dir_path = Path(__file__).parents[1]


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


# def combine_csv(folder_path):
#     os.chdir(folder_path)
#     extension = 'csv'
#     all_filenames = [i for i in glob.glob('Centar_*.{}'.format(extension))]
#     handles = [open(filename, 'r') for filename in all_filenames]
#
#     result_dataframe = pd.DataFrame()
#
#     for file in all_filenames:
#         df = pd.read_csv(file, index_col=0)
#         print()
#         if result_dataframe.empty:
#             result_dataframe = result_dataframe.append(df)
#             result_dataframe.set_index('time')
#             print(result_dataframe.head())
#         else:
#             new_df = df.loc[:, df.columns[0]].copy()
#             result_dataframe = result_dataframe.append(new_df)
#             print(result_dataframe.head())
#     print(result_dataframe.head())
#     result_dataframe.to_csv('./concat.csv', index=False, encoding='utf-8-sig')


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
    filled_data.to_csv(out_file_name, columns=[filled_data.columns[0]], index=True,  encoding='utf-8-sig')


def cut_last(file, out_file_name, last_parameter):
    df = pd.read_csv(file, index_col=0)
    df.index = pd.to_datetime(df.index)

    new_df = df.last(last_parameter)
    new_df.to_csv(out_file_name, encoding='utf-8-sig')


def generate_features(file,  out_file_name):
    df = pd.read_csv(file, index_col=0)
    df.index = pd.to_datetime(df.index)

    df['month'] = [df.index[i].month for i in range(len(df))]
    df['year'] = [df.index[i].year for i in range(len(df))]
    df.to_csv(out_file_name, encoding='utf-8-sig')

def remove_duplicates(file):
    df = get_data(file)
    new_df = df.loc[~df.index.duplicated(keep='first')]
    return new_df



