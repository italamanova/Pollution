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

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

from analysis.analyzer import check_adfuller, plot_autocorrelation, plot_distribution
from helpers.performance import measure_performance
from helpers.visualizer import plot, simple_plot

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
    plot(weekly, ylabel='%s_weekly' % col_name, title='Air pollution')


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


def my_auto_arima(file, start=None, end=None):
    data = get_data(file, start, end)

    col_name = data.columns.values[0]

    # plot(data, ylabel=col_name, title='Air pollution')
    print('Description', data[col_name].describe())

    train_size = int(len(data) * 0.8)
    test_size = len(data) - train_size
    train_copy = data[0:train_size].copy()
    test_copy = data[train_size:len(data)].copy()

    train = train_copy[col_name]
    test = test_copy[col_name]

    # result = seasonal_decompose(data, freq=24)
    # fig = result.plot()
    # plot_mpl(fig)

    stepwise_model = auto_arima(data, start_p=1, start_q=1,
                                max_p=5, max_q=5,
                                seasonal=False,
                                d=1, trace=True,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True)

    print(stepwise_model.aic())

    stepwise_model.fit(train)

    future_forecast = stepwise_model.predict(n_periods=test_size)
    print(future_forecast)

    future_forecast = pd.DataFrame(future_forecast, index=test.index, columns=['Prediction'])
    simple_plot(pd.concat([test, future_forecast], axis=1))
    rmse = measure_performance(test, future_forecast)

def pure_arima(file):
    data = get_data(file)

    col_name = data.columns.values[0]

    data = data.last('4W')

    plot(data, ylabel=col_name, title='Air pollution')
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
    rmse = measure_performance(test, future_forecast)

    future_forecast = pd.DataFrame(future_forecast, index=test.index, columns=['Prediction'])
    simple_plot(pd.concat([test, future_forecast], axis=1))
