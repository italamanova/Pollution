from statsmodels.tsa.seasonal import seasonal_decompose

from helpers.visualizer import plot
import statsmodels.api as sm

import pandas as pd

dataset = pd.read_csv('./data/Electric_Production.csv', index_col=0)
data = dataset.loc['1985-01-01':'2018-01-01']
print(data.head())

data.index = pd.to_datetime(data.index)
data.columns = ['Energy Production']

plot(data, ylabel='Energy consumption', title='Energy consumption')
print('PM10 description', data['Energy Production'].describe())
print(len(data))

result = seasonal_decompose(data, model='multiplicative')
fig = result.plot()
# plot_mpl(fig)

# stepwise_model = auto_arima(data, start_p=1, start_q=1,
#                             max_p=3, max_q=3, m=12,
#                             start_P=0, seasonal=True,
#                             d=1, D=1, trace=True,
#                             error_action='ignore',
#                             suppress_warnings=True,
#                             stepwise=True)

# print(stepwise_model.aic())
train = data.loc['1985-01-01':'2016-12-01']
test = data.loc['2017-01-01':]

stepwise_model = sm.tsa.statespace.SARIMAX(train, order=(1, 1, 1), seasonal_order=(2, 1, 2, 12))

# print(stepwise_model.aic())
model_fit = stepwise_model.fit()

print(model_fit.summary())
future_forecast = model_fit.predict('2017-01-01','2018-01-01')
print(future_forecast)

future_forecast = pd.DataFrame(future_forecast, index=test.index, columns=['Prediction'])
plot(pd.concat([test, future_forecast], axis=1))

'''
Fit ARIMA: order=(1, 1, 1) seasonal_order=(2, 1, 2, 12); AIC=1855.121, BIC=1887.033, Fit time=17.693 seconds

'''

plot(pd.concat([data,future_forecast],axis=1))
