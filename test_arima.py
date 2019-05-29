from pmdarima.arima import auto_arima

import pandas as pd
data = pd.read_csv('Electric_Production.csv',index_col=0)
data.head()
