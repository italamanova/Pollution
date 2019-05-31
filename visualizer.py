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


# data1 = pd.DataFrame([[1, 34],
#                       [2, 30],
#                       [3, 16]])
# data2 = pd.DataFrame([[1, 1],
#                       [2, 2],
#                       [3, 3]])
# plot(pd.concat([data1, data2], axis=1))
