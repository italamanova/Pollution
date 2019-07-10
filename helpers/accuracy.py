from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def measure_accuracy(test_data, predictions):
    mse = mean_squared_error(test_data, predictions)
    rmse = sqrt(mse)
    mape = mean_absolute_percentage_error(test_data, predictions)
    print('\nMSE: %.3f \nRMSE: %.3f \nMAPE: %.3f' % (mse, rmse, mape))
    return {'mse': '%.3f' % mse, 'rmse': '%.3f' % rmse, 'mape': '%.3f' % mape}


def measure_accuracy_avg_sample(test_data, predictions):
    tips = []
    pips = []
    rmse_measures = []
    rmse_results = []
    for p, t in zip(predictions, test_data):
        tips.append(t)
        pips.append(p)

        rmse = mean_squared_error(tips, pips)
        rmse_measures.append(rmse)

    first = rmse_measures[0]
    for i in range(1, len(rmse_measures)):
        result = rmse_measures[i] / i / first
        rmse_results.append(result)
    # print('RESULT', rmse_results)
    return rmse_results


def measure_accuracy_each_sample(test_data, predictions):
    rmse_measures = []
    rmse_results = []
    for p, t in zip(predictions, test_data):
        rmse = mean_squared_error([t], [p])
        rmse_measures.append(rmse)

    first = rmse_measures[0]
    for i in range(1, len(rmse_measures)):
        result = rmse_measures[i] / first
        rmse_results.append(result)
    # print('RESULT', rmse_results)
    return rmse_results
