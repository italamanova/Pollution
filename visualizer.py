import matplotlib.pyplot as plt


def plot(dataset, xlabel='Date time', ylabel='PM10', title='Time Series of PM10 by date time'):
    dataset.plot()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def scatter_plot(dataset, xlabel='Date time', ylabel='PM10', title='Time Series of PM10 by date time',
                 point_size=10):
    plt.scatter(dataset.index, dataset, s=point_size)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
