from pmdarima.arima import auto_arima
import plotly.plotly as ply
from plotly.plotly import plot_mpl
from statsmodels.tsa.seasonal import seasonal_decompose
from visualizer import plot

import pandas as pd

data = pd.read_csv('./data/Electric_Production.csv', index_col=0)
print(data.head())

data.index = pd.to_datetime(data.index)
data.columns = ['Energy Production']

plot(data, xlabel='DateTime', ylabel='Energy consumption')
print('PM10 description', data['Energy Production'].describe())
print(len(data))

# result = seasonal_decompose(data, model='multiplicative')
# fig = result.plot()
# # plot_mpl(fig)
#
stepwise_model = auto_arima(data, start_p=1, start_q=1,
                            max_p=3, max_q=3, m=12,
                            start_P=0, seasonal=True,
                            d=1, D=1, trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True)
print(stepwise_model.aic())
