import random
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from helpers.preparator import get_data
from lstm.lstm_selection.analyzer import analyze
from lstm.lstm_selection.predictor import Predictor
from lstm.lstm_selection.preparator import prepare_data

path_prepared = '%s/pollution_data/centar' % Path(__file__).parents[2]
path_to_file_prepared = '%s/Centar_PM25_prepared.csv' % path_prepared
# df, lambda_ = get_data_with_box_cox(path_to_file_prepared)
df = get_data(path_to_file_prepared)
df = df.iloc[:2000]
df_val = df[[df.columns[0]]].values.astype(float)

random.seed(7)

scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_df = scaler.fit_transform(df_val)
datas = [scaled_df, scaled_df]

n_steps_in = 24
n_steps_out = 24
batch_size = 24
is_stateful = False

test_size = batch_size
validation_size = batch_size

X, y, Xv, yv, Xt, yt = prepare_data(datas, validation_size, test_size, batch_size, n_steps_in, n_steps_out)
predictor = Predictor(X, y, Xv, yv, Xt, yt, n_steps_in, n_steps_out, batch_size, is_stateful, epochs=50)
yhat, history = predictor.predict(model_name='stacked')

analyze(scaler, X, Xt, yhat)
