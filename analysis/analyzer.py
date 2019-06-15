import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

from analysis.trend_seasonality_checker import check_polyfit, check_seasonal_decomposition
from helpers.preparator import delete_outliers, get_data
from helpers.visualizer import simple_plot


def check_adfuller(data):
    X = data[data.columns[0]].values
    result = adfuller(X)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))


def plot_autocorrelation(data):
    pyplot.figure()

    pyplot.subplot(211)
    # pyplot.axis([0, 50, -1, 1])
    plot_acf(data, ax=pyplot.gca())
    pyplot.subplot(212)
    # pyplot.axis([0, 50, -1, 1])
    plot_pacf(data, ax=pyplot.gca())
    pyplot.show()


def plot_distribution(data, col_name):
    pyplot.figure(1)
    pyplot.subplot(211)
    data[col_name].hist()
    pyplot.subplot(212)
    data[col_name].plot(kind='kde')
    pyplot.show()


def plot_average(data):
    resampled = data.resample('M').sum()
    # simple_plot(resampled, title='Air pollution')
    return resampled


def analyze(file):
    df = get_data(file)
    col_name = df.columns[0]
    new_df = df
    # new_df = df.loc['2016-01-01 00:00:00':'2017-01-01 00:00:00']
    print(len(new_df))
    # simple_plot(new_df)
    # plot_distribution(new_df, col_name)
    # df_no_outliers = delete_outliers(new_df)
    # plot_distribution(df_no_outliers, col_name)
    # simple_plot(df_no_outliers)

    check_polyfit(df)

    # reasmpled = plot_average(df)
    # check_adfuller(df)
    # check_seasonal_decomposition(df_no_outliers)
    # plot_autocorrelation(df_no_outliers)
