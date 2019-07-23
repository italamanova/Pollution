import numpy

from helpers.accuracy import measure_accuracy
from helpers.visualizer import plot_numpy_arrays
from lstm.lstm_selection.preparator import inverse_scale


def analyze(scaler, train, test, pred):
    inverse_train, inverse_test, inverse_pred = inverse_scale(scaler, train, test, pred)
    accuracy = measure_accuracy(inverse_test, inverse_pred)
    plot_numpy_arrays(inverse_train, inverse_test, inverse_pred)
