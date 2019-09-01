import matplotlib.pyplot as plt
import tikzplotlib
from scipy.stats import sem, t
from scipy import mean
import pandas as pd
import datetime
from pathlib import Path
import matplotlib.dates as mdates

parent_dir_path = Path(__file__).parents[1]


def simple_plot(dataset, xlabel='', ylabel='Value', title=''):
    dataset.plot()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_to_file(dataset, xlabel='', ylabel='', title='', out_file_name=None):
    dataset.plot()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # plt.show()
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if out_file_name:
        plt.savefig('%s/plots/%s.svg' % (parent_dir_path, out_file_name))
    else:
        plt.savefig('%s/plots/plot_%s.svg' % (parent_dir_path, now))


def scatter_plot(dataset, xlabel='', ylabel='', title='',
                 point_size=10):
    plt.scatter(dataset.index, dataset, s=point_size)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def count_confidence_interval(series):
    confidence = 0.95

    n = len(series)
    m = mean(series)
    std_err = sem(series)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)

    start = m - h
    end = m + h
    return start, end


def prepare_plot_prediction(train, test, prediction, title, df, xlabel, ylabel, date_format,
                            train_label, test_label, prediction_label):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if date_format:
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    _ = plt.xticks(rotation=90)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    ax.set_title(title)

    ax.plot(train.index, train.values, label=train_label)
    ax.plot(test.index, test.values, label=test_label)
    ax.plot(test.index, prediction, color='#3c763d', label=prediction_label)

    # std_deviation = 2 * prediction.std()
    # print(prediction.shape)
    # plt.fill_between(test.index, (prediction - 2 * std_deviation)[0], (prediction + 2 * std_deviation)[0],
    #                  color='b', alpha=.1)

    ax.legend()


def plot_prediction(train, test, prediction, title='', df=None, xlabel=None, ylabel=None, date_format=None,
                    train_label='Train', test_label='Test', prediction_label='Prediction'):
    prepare_plot_prediction(train, test, prediction, title=title, df=df, xlabel=xlabel, ylabel=ylabel,
                            date_format=date_format,
                            train_label=train_label, test_label=test_label, prediction_label=prediction_label)
    plt.show()


def plot_prediction_to_file(train, test, prediction, title='', df=None, xlabel=None, ylabel=None, date_format=None,
                            train_label='Train', test_label='Test', prediction_label='Prediction', out_file_name='123'):
    prepare_plot_prediction(train, test, prediction, title=title, df=df, xlabel=xlabel, ylabel=ylabel,
                            date_format=date_format,
                            train_label=train_label, test_label=test_label, prediction_label=prediction_label)
    plt.savefig('%s/plots/%s.svg' % (parent_dir_path, out_file_name))


def plot_errors(errors):
    mae = errors.get('each_mae')
    mape = errors.get('each_mape')

    if mae:
        plt.plot(mae)
        plt.title('MAE')
        plt.show()

    if mape:
        plt.plot(mape)
        plt.title('MAPE')
        plt.show()


def plot_numpy_arrays(train, test, prediction):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('LSTM')

    ax.plot(test, label='TEST')
    ax.plot(prediction, color='#3c763d', label='PREDICTIONS')

    ax.legend()
    plt.show()
