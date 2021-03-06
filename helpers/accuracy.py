from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from math import sqrt
import numpy as np


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def measure_accuracy(test_data, predictions):
    mse = mean_squared_error(test_data, predictions)
    rmse = sqrt(mse)
    mape = mean_absolute_percentage_error(test_data, predictions)
    mae = mean_absolute_error(test_data, predictions)
    print('\nMSE: %.3f \nRMSE: %.3f \nMAPE: %.3f \nMAE: %.3f' % (mse, rmse, mape, mae))
    return {'mse': '%.3f' % mse, 'rmse': '%.3f' % rmse, 'mape': '%.3f' % mape, 'mae': '%.3f' % mae}


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
    mae_measures = []
    for p, t in zip(predictions, test_data):
        mae = round(mean_absolute_error([t], [p]), 3)
        mae_measures.append(mae)

    return mae_measures


def measure_rmse_each_sample(test_data, predictions):
    rmse_measures = []
    try:
        for t, p in zip(test_data, predictions):
            mse = mean_squared_error([t], [p])
            rmse = round(sqrt(mse), 3)
            rmse_measures.append(rmse)
    except Exception as e:
        print('EXCEPTION', e)
    return rmse_measures


# def measure_accuracy_each_sample(test_data, predictions):
#     accuracy_measures = []
#     accuracy_results = []
#     for p, t in zip(predictions, test_data):
#         accuracy = mean_absolute_percentage_error([t], [p])
#         accuracy_measures.append(accuracy)
#
#     for i in range(1, len(accuracy_measures)):
#         result = accuracy_measures[i]
#         accuracy_results.append(result)
#
#     return accuracy_results


def accuracy_evaluation(test, predictions):
    accuracy = measure_accuracy(test, predictions)
    each_sample_accuracy = measure_rmse_each_sample(test, predictions)
    result = {'accuracy': accuracy,
              'each_sample_accuracy': each_sample_accuracy}
    # result = {'accuracy': accuracy
    #           }
    return result
