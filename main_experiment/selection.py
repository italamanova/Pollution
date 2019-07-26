import time

import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

from es.es import exponential_smoothing
from es.es_grid_search import es_grid_search
from helpers.accuracy import measure_accuracy, measure_accuracy_each_sample, measure_rmse_each_sample, \
    accuracy_evaluation
from helpers.saver import print_to_file, update_and_print_to_file
from helpers.visualizer import plot_prediction, plot_errors
from myarima.arima import predict_auto_arima


def predict_on_train_es(train, test, lambda_):
    reversed_train, reversed_test, reversed_pred, model_params = exponential_smoothing(train, test, lambda_)
    result_json = {}
    result_json.update(model_params)
    accuracy = accuracy_evaluation(reversed_pred.values, reversed_test.values)
    result_json.update(accuracy)
    return result_json


def predict_on_train_grid_search_es(train, test, lambda_):
    result_json = es_grid_search(train, test, lambda_)
    return result_json


def predict_on_train_arima(train, test, lambda_):
    reversed_train, reversed_test, reversed_pred, model_params = predict_auto_arima(train, test, lambda_,
                                                                                    start_p=0, max_p=6,
                                                                                    max_d=2,
                                                                                    start_q=0, max_q=5,
                                                                                    start_P=0, max_P=3,
                                                                                    max_D=1,
                                                                                    start_Q=0, max_Q=2,
                                                                                    seasonal=True, m=24,
                                                                                    stepwise=True,
                                                                                    information_criterion='aic'
                                                                                    )
    # plot_prediction(reversed_train, reversed_test, reversed_pred, title='ARIMA')
    result_json = {}
    result_json.update(model_params)
    accuracy = accuracy_evaluation(reversed_pred.values, reversed_test.values)
    result_json.update(accuracy)
    return result_json


def run_select(df, train_window, step, test_window, rolling_window, lambda_, out_file, method_name='es'):
    train_df = df[:train_window]
    result_json = {'train_start': train_df.index[0],
                   'train_window': train_window,
                   'step': step,
                   'test_window': test_window,
                   'rolling_window': rolling_window,
                   'lambda': lambda_[0],
                   'results': []}

    for i in range(0, rolling_window, step):
        start_time = time.time()
        current_result = {}
        current_train = df.iloc[i:i + train_window]
        current_test = df.iloc[i + train_window:i + train_window + test_window]

        current_result.update({
            'train': '%s - %s' % (current_train.index[0], current_train.index[-1]),
            'test': '%s - %s' % (current_test.index[0], current_test.index[-1]),
            'step': i
        })
        if method_name == 'es':
            # prediction_result = predict_on_train_es(current_train, current_test, lambda_)
            prediction_result = predict_on_train_grid_search_es(current_train, current_test, lambda_)
        if method_name == 'arima':
            prediction_result = predict_on_train_arima(current_train, current_test, lambda_)

        current_result.update(prediction_result)
        end_time = time.time()
        current_result.update({'time': end_time - start_time})
        result_json['results'].append(current_result)

        print_to_file(out_file, result_json)
