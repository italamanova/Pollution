import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

from es.es import exponential_smoothing
from helpers.accuracy import measure_accuracy, measure_accuracy_each_sample
from helpers.visualizer import plot_prediction


def predict_on_train(df, train, test, method_name):
    if method_name == 'es':
        predictions = exponential_smoothing(train, test)
    if method_name == 'arima':
        predictions = exponential_smoothing(train, test)
    plot_prediction(train, test, predictions, title=method_name)
    return predictions


def accuracy_evaluation(test, predictions):
    measure_accuracy(test, predictions)
    each_sample_accuracy = measure_accuracy_each_sample(test, predictions)
    plt.plot(each_sample_accuracy)
    plt.show()


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
