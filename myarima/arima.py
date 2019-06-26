from pathlib import Path

import pandas as pd
import seaborn
from math import sqrt
from plotly.offline import plot_mpl
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.stattools import adfuller
import numpy as np

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

from analysis.analyzer import plot_distribution
from analysis.trend_seasonality_checker import check_seasonal_decomposition
from helpers.decorators import timeit
from helpers.accuracy import measure_accuracy
from helpers.preparator import cut_dataframe
from helpers.visualizer import plot_to_file, simple_plot, plot_prediction

parent_dir_path = Path(__file__).parents[1]


def check_seasonality(file, start_date, end_date):
    series = Series.from_csv(file, header=0)
    series = series.loc[start_date:end_date]
    resample = series.resample('D')
    monthly_mean = resample.mean()
    print(monthly_mean)
    monthly_mean.plot()
    pyplot.show()


def get_data(file, start=None, end=None):
    df = pd.read_csv(file, index_col=0)
    df.index = pd.to_datetime(df.index)

    if start and end:
        data = df.loc[start:end]
    else:
        data = df
    # data.fillna(method='ffill', inplace=True)
    return data


def plot_average(data, col_name):
    weekly = data.resample('M').sum()
    plot_to_file(weekly, ylabel='%s_weekly' % col_name, title='Air pollution')


def plot_rolling_average(data, col_name):
    rolling = data.rolling(window=5040)
    rolling_mean = rolling.mean()
    data.plot()
    rolling_mean.plot(color='red')
    pyplot.show()


def plot_boxplot(data):
    data.boxplot()
    pyplot.show()


def analyze_data(file, start=None, end=None):
    data = get_data(file, start, end)
    col_name = data.columns.values[0]

    # plot_average(data, col_name)
    # plot_rolling_average(data, col_name)

    # data = data.last('4W')
    plot_distribution(data, col_name)
    #
    # check_adfuller(data, col_name)
    #
    # plot_autocorrelation(data)
    # plot_boxplot(data)


@timeit
def my_auto_arima(file, start=None, end=None):
    data = get_data(file, start, end)
    print(data[data.isna().any(axis=1)])

    # start = '2018-01-10 00:00:00'
    # end = '2018-02-10 00:00:00'
    #
    # data = cut_dataframe(data, start, end)
    # data = data.last('2W')

    col_name = data.columns.values[0]

    # plot(data, ylabel=col_name, title='Air pollution')
    print('Description', data[col_name].describe())

    train_size = int(len(data) * 0.8)
    test_size = len(data) - train_size
    train_copy = data[0:train_size].copy()
    test_copy = data[train_size:len(data)].copy()

    train = train_copy[col_name]
    test = test_copy[col_name]

    # check_seasonal_decomposition(data)

    stepwise_model = auto_arima(train, start_p=0, max_p=3,
                                start_q=0, max_q=3,
                                start_P=0, max_P=2,
                                start_Q=0, max_Q=2,
                                seasonal=True, m=24,
                                # d=1,
                                # D=0, max_D=1,
                                max_order=None,
                                trace=True,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True  # True
                                )

    print(stepwise_model.aic())
    print(stepwise_model.__dict__)

    stepwise_model.fit(train)

    future_forecast = stepwise_model.predict(n_periods=test_size)
    print(future_forecast)

    future_forecast = pd.DataFrame(future_forecast, index=test.index, columns=['Prediction'])
    plot_prediction(train, test, future_forecast, title='ARIMA')
    measure_accuracy(test, future_forecast)


def pure_arima(file):
    data = get_data(file)

    col_name = data.columns.values[0]

    # start = '2018-01-10 00:00:00'
    # end = '2018-02-10 00:00:00'
    #
    # data = cut_dataframe(data, start, end)

    plot_to_file(data, ylabel=col_name, title='Air pollution')
    print('Description', data[col_name].describe())

    train_size = int(len(data) * 0.8)
    train_copy = data[0:train_size].copy()
    test_copy = data[train_size:len(data)].copy()

    train = train_copy[col_name]
    test = test_copy[col_name]

    model = ARIMA(train, order=(5, 1, 1))
    model_fit = model.fit(disp=0)
    future_forecast = model_fit.forecast(steps=len(test))[0]
    # observation
    # report performance

    future_forecast = pd.DataFrame(future_forecast, index=test.index, columns=['Prediction'])
    plot_prediction(train, test, future_forecast)
