from warnings import catch_warnings, filterwarnings

from es.es import exponential_smoothing
from helpers.accuracy import measure_accuracy, measure_rmse_each_sample, accuracy_evaluation


def get_grid_search_configs():
    models = list()
    # define config lists
    trend = ['add', 'mul', None]
    damped = [True, False]
    seasonality = ['add', 'mul', None]
    for t in trend:
        for d in damped:
            for s in seasonality:
                cfg = [t, d, s]
                models.append(cfg)
    return models


def build_model_with_config(config, train, test, lambda_):
    reversed_train, reversed_test, reversed_pred, model_params = exponential_smoothing(train, test, lambda_,
                                                                                       trend=config[0],
                                                                                       seasonal=config[2],
                                                                                       seasonal_periods=24,
                                                                                       damped=config[1])
    accuracy = accuracy_evaluation(reversed_test.values, reversed_pred.values)

    result_params = dict()
    result_params.update(model_params)
    result_params.update(accuracy)
    print()
    print(model_params)
    print()
    return result_params


def select_best_model(model_list):
    best_model = min(model_list, key=lambda el: float(el['accuracy']['rmse']))
    return best_model


def es_grid_search(train, test, lambda_):
    configs = get_grid_search_configs()
    result_list = []
    for config in configs:
        try:
            with catch_warnings():
                filterwarnings("ignore")
                result_params = build_model_with_config(config, train, test, lambda_)
                result_list.append(result_params)
        except Exception as e:
            pass
            # result_list.append({'ERROR':config})

    best_model = select_best_model(result_list)
    return best_model
