import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

from helpers.visualizer import simple_plot


def get_data(file):
    df = pd.read_csv(file, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df


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


def plot_average(data):
    resampled = data.resample('M').sum()
    # simple_plot(resampled, title='Air pollution')
    return resampled


def analyze_average(file):
    df = get_data(file)
    # reasmpled = plot_average(df)
    check_adfuller(df)
    # plot_autocorrelation(df)
