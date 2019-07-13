import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

from es.es import exponential_smoothing
from helpers.accuracy import measure_accuracy, measure_accuracy_each_sample, measure_mae_each_sample
from helpers.saver import print_to_file
from helpers.visualizer import plot_prediction, plot_errors
from myarima.arima import predict_auto_arima


def predict_on_train(train, test, lambda_, method_name):
    print('current_train.shape', train.shape)
    if method_name == 'es':
        reversed_train, reversed_test, reversed_pred = exponential_smoothing(train, test, lambda_)
        plot_title = 'Exponential Smoothing'
    if method_name == 'arima':
        # TODO add arima model paramms!
        reversed_train, reversed_test, reversed_pred = predict_auto_arima(train, test, lambda_)
        plot_title = 'ARIMA'
    plot_prediction(reversed_train, reversed_test, reversed_pred, title=plot_title)
    return reversed_train, reversed_test, reversed_pred


def accuracy_evaluation(test, predictions):
    accuracy = measure_accuracy(test, predictions)
    errors = {'each_mape': measure_accuracy_each_sample(test, predictions),
              'each_mae': measure_mae_each_sample(test, predictions)}
    plot_errors(errors)
    accuracy.update(errors)
    return accuracy


def select_train(df, train_start_length, step, test_length, lambda_, method_name='es'):
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

        reversed_train, reversed_test, reversed_pred = predict_on_train(current_train, current_test, lambda_,
                                                                        method_name)
        accuracy = accuracy_evaluation(reversed_pred.values, reversed_test.values)
        current_result.update(accuracy)

        result_json['results'].append(current_result)
        i += 1
        train_length += step
    print_to_file('koko.json', result_json)
