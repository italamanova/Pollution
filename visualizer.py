import matplotlib.pyplot as plt
import pandas as pd


def plot(dataset, xlabel='DateTime', ylabel='Value', title='Plot'):
    dataset.plot()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def scatter_plot(dataset, xlabel='DateTime', ylabel='Value', title='Plot',
                 point_size=10):
    plt.scatter(dataset.index, dataset, s=point_size)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
