import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

from es.es import exponential_smoothing
from es.es_grid_search import es_grid_search
from helpers.accuracy import measure_accuracy, measure_accuracy_each_sample, measure_mae_each_sample
from helpers.saver import print_to_file
from helpers.visualizer import plot_prediction, plot_errors
from myarima.arima import predict_auto_arima


def accuracy_evaluation(test, predictions):
    accuracy = measure_accuracy(test, predictions)
    errors = {'each_mape': measure_accuracy_each_sample(test, predictions)}
    # plot_errors(errors)
    # accuracy.update(errors)
    return accuracy


def predict_on_train_es(train, test, lambda_):
    reversed_train, reversed_test, reversed_pred, model_params = exponential_smoothing(train, test, lambda_)
    # plot_prediction(reversed_train, reversed_test, reversed_pred, title='Exponential Smoothing')
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
    plot_prediction(reversed_train, reversed_test, reversed_pred, title='ARIMA')
    result_json = {}
    result_json.update(model_params)
    accuracy = accuracy_evaluation(reversed_pred.values, reversed_test.values)
    result_json.update(accuracy)
    return result_json


def predict_on_train_lstm(train, test, lambda_):
    pass


def select_train(df, train_start_length, step, test_length, lambda_, method_name='es', out_file_name='1'):
    result_json = {'train_start_length': train_start_length,
                   'step': step,
                   # 'from': df.index[0],
                   # 'to': df.index[-1],
                   'lambda': lambda_[0],
                   'results': []}
    train_length = train_start_length
    i = 0
    while train_length <= len(df) - test_length:
        current_result = {}
        current_train = df.iloc[:train_length]
        current_test = df.iloc[train_length:train_length + test_length]

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
        # if method_name == 'lstm':
        #     prediction_result = predict_on_train_lstm(current_train, current_test, lambda_)

        current_result.update(prediction_result)
        result_json['results'].append(current_result)
        i += 1
        train_length += step
    print_to_file('%s_%s.json' % (method_name, out_file_name), result_json)
