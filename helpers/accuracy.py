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
    return rmse
