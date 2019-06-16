import pandas as pd
import statsmodels.formula.api as smf
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


def get_resampled(df, period_name):
    resampled = df.resample(period_name).sum()
    return resampled


# def polynomial_regression(df):
#     col_name = df.columns[0]
#     X = [i % 365 for i in range(0, len(df))]
#     y = df[col_name].values
#
#     weights = np.polyfit(X, y, 20)
#
#     model = np.poly1d(weights)
#     results = smf.ols(formula='y ~ model(X)', data=df).fit()
#
#     print(results.summary())


def analyze(file):
    df = get_data(file)
    col_name = df.columns[0]
    period_name = 'D'
    degree = 1
    # new_df = df.loc['2016-01-01 00:00:00':'2017-01-01 00:00:00']
    # simple_plot(df)
    # plot_distribution(df, col_name)

    resampled = get_resampled(df, period_name)
    check_polyfit(resampled, degree)

    # reasmpled = plot_average(df)
    # check_adfuller(df)
    # check_seasonal_decomposition(df_no_outliers)
    # plot_autocorrelation(df)
