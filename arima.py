import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import plotly
import plotly.plotly as ply
from mpl_toolkits.mplot3d import Axes3D
from plotly.plotly import plot_mpl
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
from pmdarima.arima import auto_arima

plotly.tools.set_credentials_file(username='talamash', api_key='NVqgaGN3OpMYcqXncLOw')

# prepare data

# df = pd.read_csv('pollution_backup.csv')
# df.drop('SystemCodeNumber', axis=1, inplace=True)
# df.to_csv('pollution.csv', index=False)
# from helpers import detect_outlier, plotMovingAverage

data = pd.read_csv('./data/pollution2.csv', index_col=0)
# data[data <= 0] = 0.01
data.index = pd.to_datetime(data.index)
data.pm = data['PM10']

# print(data.info())
print('PM10 description', data['PM10'].describe())
print('Max value of PM10', data.loc[(data['PM10'] == 822.699000)])

'''
Data visualization
'''

# data.plot()
# plt.xlabel('Date time')
# plt.ylabel('PM10')
# plt.title('Time Series of PM10 by date time')
# plt.show()
#
# # Check data with scatter plot
# plt.scatter(data.index, data, s=10)
# plt.show()


'''
Removing outliers
'''
print('OUTLIERS')
# filtering_rule_1 = (data.pm.median() - data.pm).abs() > 0.3
# print('OUTLIERS', len(data[~filtering_rule_1]))

lower_bound = .25
upper_bound = .75
quant_df = data.quantile([lower_bound, upper_bound])

filtering_rule_2 = data.apply(
    lambda x: (x < quant_df.loc[lower_bound, x.name]) | (x > quant_df.loc[upper_bound, x.name]), axis=0)

dataframe = data[~(filtering_rule_2).any(axis=1)]

dataframe.plot()
plt.xlabel('Date time')
plt.ylabel('PM10')
plt.title('After Filtering Time Series of PM10 by date time')
plt.show()

'''
ARIMA seasonal decomposition
'''

# result = seasonal_decompose(dataframe, model='multiplicative', freq=30)
# fig = result.plot()
# plot_mpl(fig)

# building trend
# plotMovingAverage(dataframe, 24)
# plotMovingAverage(dataframe, 24*7)

# stepwise_model = auto_arima(data, start_p=1, start_q=1,
#                            max_p=3, max_q=3, m=12,
#                            start_P=0, seasonal=True,
#                            d=1, D=1, trace=True,
#                            error_action='ignore',
#                            suppress_warnings=True,
#                            stepwise=True)
# print(stepwise_model.aic())


