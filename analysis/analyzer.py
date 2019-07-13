from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import pyplot
from pandas.plotting import lag_plot, autocorrelation_plot
from scipy.stats import variation
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.iolib import SimpleTable
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller, kpss

from analysis.trend_seasonality_checker import check_seasonal_decomposition, check_polyfit
from helpers.converter import get_resampled
from helpers.preparator import delete_outliers, get_data
from helpers.visualizer import simple_plot
from stationarity.differencing_checker import fourier

parent_dir_path = Path(__file__).parents[1]


def describe_data(df):
    series = df[df.columns[0]]
    print('Description', series.describe())
    print('Coefficient of variation(V) = %f' % variation(series))
    # =======
    # Calculates the Jarque-Bera test for normality

    row = [u'JB', u'p-value', u'skew', u'kurtosis']
    jb_test = jarque_bera(series)
    a = np.vstack([jb_test])
    r = SimpleTable(a, row)
    print(r)


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
    # pyplot.savefig('%s/plots/%s_distribution.png' % (parent_dir_path, col_name))
    pyplot.show()


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
    period_name = 'M'
    degree = 1

    # simple_plot(df)
    df = df.iloc[24*365:24*365+24*30]
    # df = df.loc['2015-02-01 00:00:00':'2015-07-01 00:00:00']
    # simple_plot(df)
    # plot_distribution(df, col_name)
    # df = get_resampled(df, period_name)

    # scatter_lag_plot(df, 24)
    # describe_data(df)
    # my_autocorrelation_plot(df)
    # fourier(df[col_name], df.index, 24)

    # check_polyfit(df, degree)

    # check_adfuller(df)
    # check_kpss(df)
    simple_plot(df)
    check_seasonal_decomposition(df)
    # plot_autocorrelation(df)


def scatter_lag_plot(df, lag=1):
    series = df[df.columns[0]]
    lag_plot(series, lag=lag, s=5)
    pyplot.show()


def my_autocorrelation_plot(df):
    # series = df[df.columns[0]]
    autocorrelation_plot(df)
    pyplot.show()
