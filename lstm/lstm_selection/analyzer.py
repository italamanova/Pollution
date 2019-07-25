from helpers.accuracy import measure_accuracy, measure_rmse_each_sample
from helpers.saver import print_to_file, update_and_print_to_file


def create_json(reversed_train, reversed_test, reversed_pred,
                model_config, lambda_, df):
    sum_steps = model_config['n_steps_in'] + model_config['n_steps_out']
    train_df = df[:len(df) - sum_steps]
    print(train_df.index[0])
    result_json = {'train_start': '%s' % (train_df.index[0]),
                   'train_end': '%s' % (train_df.index[-1]),
                   'train_window': len(reversed_train),
                   'lambda': lambda_[0],
                   'model': model_config,
                   'results': []}

    for i in range(0, len(reversed_test)):
        out_json = {
            'step_number': i
        }
        accuracy = measure_accuracy(reversed_test[i], reversed_pred[i])
        each_sample_accuracy = measure_rmse_each_sample(reversed_test[i], reversed_pred[i])
        out_json['accuracy'] = accuracy
        out_json['each_sample_accuracy'] = each_sample_accuracy

        result_json['results'].append(out_json)

    return result_json


def analyze(out_file, reversed_train, reversed_test, reversed_pred,
            model_config, lambda_, df):
    out_json = create_json(reversed_train, reversed_test, reversed_pred,
                           model_config, lambda_, df)
    print_to_file(out_file, out_json)


def analyze_multiple(out_file, reversed_train, reversed_test, reversed_pred,
                     model_config, lambda_, df):
    out_json = create_json(reversed_train, reversed_test, reversed_pred,
                           model_config, lambda_, df)
    update_and_print_to_file(out_file, out_json)
