from pathlib import Path

import pandas as pd
from plotly.offline import plot_mpl
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas import Series
from matplotlib import pyplot

import statsmodels.api as sm

from helpers.visualizer import plot, simple_plot

parent_dir_path = Path(__file__).parents[1]


def check_seasonality(file, start_date,  end_date):
    series = Series.from_csv(file, header=0)
    series = series.loc[start_date:end_date]
    resample = series.resample('D')
    monthly_mean = resample.mean()
    print(monthly_mean)
    monthly_mean.plot()
    pyplot.show()


def arima(file, start, end):
    df = pd.read_csv(file, index_col=0)

    df.index = pd.to_datetime(df.index)
    # df = df.asfreq(freq='H')
    data = df.loc[start:end]
    data = data.fillna(0.01)
    print(data.index.freq)

    col_name = data.columns.values[0]
    count_nan = data.isnull().sum(axis=0)
    print('count_nan', count_nan)

    # plot(data, ylabel=col_name, title='Air pollution')
    print('Description', data[col_name].describe())
    print(len(data))

    result = seasonal_decompose(data, freq=24*30)
    fig = result.plot()
    plot_mpl(fig)

    # stepwise_model = auto_arima(data, start_p=1, start_q=1,
    #                             max_p=3, max_q=3, m=24*7*52,
    #                             start_P=0, seasonal=True,
    #                             d=1, D=1, trace=True,
    #                             error_action='ignore',
    #                             suppress_warnings=True,
    #                             stepwise=True)

    stepwise_model = auto_arima(data, start_p=1, start_q=1,
                                max_p=3, max_q=3, m=52,
                                start_P=0, seasonal=True,
                                d=1, D=1, trace=True,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True)

    print(stepwise_model.aic())

    # print(stepwise_model.aic())
    train = data.loc['2017-12-01 00:00:00':'2018-01-01 00:00:00']
    test = data.loc['2018-01-01 00:00:00':]
    # print('test', len(test))
    stepwise_model.fit(train)

    future_forecast = stepwise_model.predict(n_periods=25)
    print(future_forecast)

    future_forecast = pd.DataFrame(future_forecast, index=test.index, columns=['Prediction'])
    simple_plot(pd.concat([test, future_forecast], axis=1))
