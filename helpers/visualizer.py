import matplotlib.pyplot as plt
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


def plot_to_file(dataset, xlabel='DateTime', ylabel='Value', title='', out_file_name=None):
    dataset.plot()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # plt.show()
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if out_file_name:
        plt.savefig('%s/plots/%s.png' % (parent_dir_path, out_file_name))
    else:
        plt.savefig('%s/plots/plot_%s.png' % (parent_dir_path, now))


def scatter_plot(dataset, xlabel='DateTime', ylabel='Value', title='Plot',
                 point_size=10):
    plt.scatter(dataset.index, dataset, s=point_size)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_prediction(train, test, prediction, title='', xlabel=None, ylabel=None, date_format=None,
                    train_label='Train', test_label='Test', prediction_label='Prediction'):
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

    ax.legend()
    plt.show()
