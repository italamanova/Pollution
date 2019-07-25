from helpers.accuracy import measure_accuracy, measure_rmse_each_sample
from helpers.saver import print_to_file, update_and_print_to_file


def create_json(reversed_train, reversed_test, reversed_pred,
                model_config, train_df, test_df):
    result_json = {'train_period': ' %s - %s' % (train_df.index[0], train_df.index[-1]),
                   'test_period': '%s - %s' % (test_df.index[0], test_df.index[-1]),
                   'model_config': model_config,
                   'results': []}

    for i in range(0, len(reversed_test)):
        out_json = {
            'period_number': i
        }
        accuracy = measure_accuracy(reversed_test[i], reversed_pred[i])
        each_sample_accuracy = measure_rmse_each_sample(reversed_test[i], reversed_pred[i])
        out_json['accuracy'] = accuracy
        out_json['each_sample_rmse'] = each_sample_accuracy

        result_json['results'].append(out_json)

    return result_json


def analyze(out_file, reversed_train, reversed_test, reversed_pred,
            model_config, train_df, test_df):
    out_json = create_json(reversed_train, reversed_test, reversed_pred,
                           model_config, train_df, test_df)
    print_to_file(out_file, out_json)


def analyze_multiple(out_file, reversed_train, reversed_test, reversed_pred,
                     model_config, train_df, test_df):
    out_json = create_json(reversed_train, reversed_test, reversed_pred,
                           model_config, train_df, test_df)
    update_and_print_to_file(out_file, out_json)
