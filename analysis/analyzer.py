from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss

from analysis.trend_seasonality_checker import check_polyfit, check_seasonal_decomposition
from helpers.preparator import delete_outliers, get_data
from helpers.visualizer import simple_plot

parent_dir_path = Path(__file__).parents[1]


def check_adfuller(data):
    series = data[data.columns[0]].values
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(series, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4],
                         index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


def check_kpss(data):
    print('Results of KPSS Test:')
    series = data[data.columns[0]].values
    kpsstest = kpss(series, regression='c')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value
    print(kpss_output)


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
    pyplot.savefig('%s/plots/%s_distribution.png' % (parent_dir_path, col_name))


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

    # df = df.loc['2014-11-07 00:00:00':'2014-11-10 00:00:00']
    # simple_plot(df)
    # simple_plot(df)
    # plot_distribution(df, col_name)

    df = get_resampled(df, period_name)
    # check_polyfit(df, degree)

    # check_adfuller(df)
    # check_kpss(df)
    check_seasonal_decomposition(df)
    plot_autocorrelation(df)
