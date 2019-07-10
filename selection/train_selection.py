from pathlib import Path

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from es.es import exponential_smoothing
from helpers.accuracy import measure_accuracy
from helpers.preparator import get_data
from helpers.visualizer import plot_prediction


def get_train_test(series):
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(series):
        print("TRAIN:", len(train_index), "TEST:", len(test_index))
        train, test = df.iloc[train_index], df.iloc[test_index]
        # print(train, test)


def predict_on_train(df, train, test):
    predictions = exponential_smoothing(train, test)
    plot_prediction(train, test, predictions, title='Exponential Smoothing')
    return predictions


def accuracy_evaluation(test, predictions):
    measure_accuracy(test, predictions)


def select_train(df, train_start_length, step, test_length):
    train_length = train_start_length
    i = 0
    while train_length <= len(df) - test_length:
        print(i)
        current_train = df.iloc[:train_length]
        current_test = df.iloc[train_length:train_length + test_length]
        predictions = predict_on_train(df, current_train, current_test)
        accuracy_evaluation(predictions, current_test)
        i += 1
        train_length += step


path_prepared = '%s/pollution_data/centar' % Path(__file__).parents[1]
path_to_file_prepared = '%s/Centar_PM25_prepared.csv' % path_prepared
df = get_data(path_to_file_prepared).iloc[:50]
# series = df[df.columns[0]].values
# get_train_test(series)
select_train(df, 30, 1, 10)
