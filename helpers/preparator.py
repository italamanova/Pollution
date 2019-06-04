import os
import glob
import pandas as pd
from pathlib import Path

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


def cut_csv(file, start='2008-01-01 00:00:00', end='2018-03-09 00:00:00'):
    filename_w_ext = os.path.basename(file)
    filename, file_extension = os.path.splitext(filename_w_ext)

    df = pd.read_csv(file, index_col=0)
    new_df = df.loc[start:end]
    new_file_name = '%s/pollution_data/cut_data/%s_%s_%s.csv' % (parent_dir_path, filename, start[:-9], end[:-9])
    new_df.to_csv(new_file_name, encoding='utf-8-sig')


def remove_outliers(dataset):
    lower_bound = .25
    upper_bound = .75
    quant_df = dataset.quantile([lower_bound, upper_bound])

    filtering_rule_2 = dataset.apply(
        lambda x: (x < quant_df.loc[lower_bound, x.name]) | (x > quant_df.loc[upper_bound, x.name]), axis=0)

    dataframe = dataset[~(filtering_rule_2).any(axis=1)]
    return dataframe
