import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
# import plotly
# import plotly.plotly as ply
# from mpl_toolkits.mplot3d import Axes3D
# from plotly.plotly import plot_mpl
# from sklearn.cluster import KMeans
# from statsmodels.tsa.seasonal import seasonal_decompose
# import seaborn as sns
from pmdarima.arima import auto_arima
import matplotlib.pyplot as plt
import pickle


def simple_plot(dataset, xlabel='DateTime', ylabel='Value', title='Plot'):
    dataset.plot()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


# prepare data

data = pd.read_csv('5e.csv', index_col=0)
tstart = '2017-02-01 00:00:00'
tend = '2017-02-06 23:00:00'
fstart = '2017-02-07 00:00:00'
fend = '2017-02-07 23:00:00'
train = data.loc[tstart:tend]
test = data.loc[fstart:fend]

model = auto_arima(train, start_p=0, max_p=3,
                   start_q=0, max_q=3,
                   start_P=0, max_P=2,
                   start_Q=0, max_Q=2,
                   seasonal=True, m=24,
                   # d=1,
                   # D=0, max_D=1,
                   max_order=None,
                   trace=True,
                   error_action='ignore',
                   suppress_warnings=True,
                   stepwise=False  # True
                   )
with open('arima.pkl', 'wb') as pkl:
    pickle.dump(model, pkl)
'''
# Or maybe joblib tickles your fancy
from sklearn.externals import joblib
joblib.dump(arima, 'arima.pkl')
joblib_preds = joblib.load('arima.pkl').predict(n_periods=5)
'''

# Now read it back and make a prediction
# with open('arima.pkl', 'rb') as pkl:
#    pickle_preds = pickle.load(pkl).predict(n_periods=5)

print('best aic=%f' % model.aic())
print(model.order)
print(model.seasonal_order)
res = model.fit(train)
forecast = model.predict(n_periods=len(test))

predict = pd.DataFrame(forecast, index=test.index, columns=['Prediction'])

# pd.concat([test,future_forecast],axis=1).iplot()
simple_plot(pd.concat([test, predict], axis=1))
# print(res.summary())
# print(stepwise_model.aic())
