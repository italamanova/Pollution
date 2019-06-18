import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt

from analysis.analyzer import get_data
from helpers.performance import measure_performance


def exponential_smoothing(file):
    data = get_data(file)
    col_name = data.columns.values[0]

    train_size = int(len(data) * 0.9)
    train_copy = data[0:train_size].copy()
    test_copy = data[train_size:len(data)].copy()

    train = train_copy[col_name]
    test = test_copy[col_name]
    # print(type(test.index[0]))

    # model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=24)
    # model = ExponentialSmoothing(train, trend='add')
    # model = Holt(train)
    model2 = ExponentialSmoothing(train, trend="add")
    # fit = model.fit(smoothing_level=0.2, smoothing_slope=0.8, optimized=False)
    # print('model 1', model.params)

    # pred = fit.forecast(len(test))
    fit2 = model2.fit()
    print('model 2', fit2.__dict__)
    pred2 = fit2.forecast(len(test))

    # sse1 = np.sqrt(np.mean(np.square(test.values - pred.values)))
    sse2 = np.sqrt(np.mean(np.square(test.values - pred2.values)))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train.index[60:], train.values[60:])
    ax.plot(test.index, test.values, label='truth')
    # ax.plot(test.index, pred, color='#008000',
    #         label="w/o damping (RMSE={:0.2f}, AIC={:0.2f})".format(sse1, fit.aic))
    ax.plot(test.index, pred2, linestyle='--', color='#3c763d',
            label="without tuning (RMSE={:0.2f}, AIC={:0.2f})".format(sse2, fit2.aic))
    ax.legend()
    ax.set_title("Holt's Seasonal Smoothing")
    plt.show()

    measure_performance(test, pred2)
