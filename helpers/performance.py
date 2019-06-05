from sklearn.metrics import mean_squared_error
from math import sqrt


def measure_performance(test_data, predictions):
    mse = mean_squared_error(test_data, predictions)
    rmse = sqrt(mse)
    print('RMSE: %.3f' % rmse)
