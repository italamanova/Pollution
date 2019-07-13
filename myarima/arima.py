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
from helpers.preparator import cut_dataframe, get_data_with_box_cox, get_data, reverse_box_cox
from helpers.visualizer import plot_to_file, simple_plot, plot_prediction

parent_dir_path = Path(__file__).parents[1]


def predict_auto_arima(train, test, lambda_,
                       start_p, max_p, start_q, max_q, max_d,
                       start_P, max_P, start_Q, max_Q, max_D,
                       m, information_criterion,
                       max_order=10,
                       d=None, D=None,
                       method=None, trend='c', solver='lbfgs',
                       suppress_warnings=True, error_action='warn', trace=True,
                       stepwise=False, seasonal=True, n_jobs=1):

    _model = auto_arima(train, start_p=start_p, max_p=max_p,
                        d=d, max_d=max_d,
                        start_q=start_q, max_q=max_q,
                        start_P=start_P, max_P=max_P,
                        D=D, max_D=max_D,
                        start_Q=start_Q, max_Q=max_Q,
                        seasonal=seasonal, m=m,
                        max_order=max_order,
                        trace=trace,
                        error_action=error_action,
                        suppress_warnings=suppress_warnings,
                        stepwise=stepwise)

    aic = _model.aic()

    _model.fit(train)

    pred = _model.predict(n_periods=len(test))
    reversed_train, reversed_test, reversed_pred = reverse_box_cox(train, test, pred, lambda_)

    return reversed_train, reversed_test, reversed_pred


@timeit
def my_auto_arima(file, test_size, start_p, max_p, start_q, max_q, max_d,
                  start_P, max_P, start_Q, max_Q, max_D,
                  m, information_criterion,
                  max_order=10,
                  d=None, D=None,
                  method=None, trend='c', solver='lbfgs',
                  suppress_warnings=True, error_action='warn', trace=True,
                  stepwise=False, seasonal=True, n_jobs=1):
    df, lambda_ = get_data_with_box_cox(file)

    col_name = df.columns[0]
    train_size = len(df) - test_size

    train_copy = df[0:train_size].copy()
    test_copy = df[train_size:len(df)].copy()

    train = train_copy[col_name]
    test = test_copy[col_name]

    reversed_train, reversed_test, reversed_pred = predict_auto_arima(train, test, lambda_,
                                                                      start_p=start_p, max_p=max_p,
                                                                      d=d, max_d=max_d,
                                                                      start_q=start_q, max_q=max_q,
                                                                      start_P=start_P, max_P=max_P,
                                                                      D=D, max_D=max_D,
                                                                      start_Q=start_Q, max_Q=max_Q,
                                                                      seasonal=seasonal, m=m,
                                                                      max_order=max_order,
                                                                      trace=trace,
                                                                      error_action=error_action,
                                                                      suppress_warnings=suppress_warnings,
                                                                      stepwise=stepwise)

    plot_prediction(reversed_train, reversed_test, reversed_pred, title='ARIMA')
    measure_accuracy(reversed_test, reversed_pred)


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
    plot_prediction(train, test, future_forecast, df=data)
