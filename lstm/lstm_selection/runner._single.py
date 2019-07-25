from pathlib import Path

from sklearn.preprocessing import MinMaxScaler

from helpers.preparator import get_data_with_box_cox
from lstm.lstm_selection.analyzer import analyze
from lstm.lstm_selection.processor import process_lstm

path_prepared = '%s/data/centar' % Path(__file__).parents[2]
path_to_file_prepared = '%s/Centar_PM25_prepared.csv' % path_prepared

path_out = '%s/data/main_experiment/lstm_experiment_results' % Path(__file__).parents[2]
m_out_file = '%s/%s.json' % (path_out, 'lstm_rmse')

df, lambda_ = get_data_with_box_cox(path_to_file_prepared)
# df = get_data(path_to_file_prepared)
df = df.iloc[:480]
df_val = df[[df.columns[0]]].values.astype(float)

scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_df = scaler.fit_transform(df_val)
datas = [scaled_df, scaled_df]

n_steps_in = 24
n_steps_out = 24
batch_size = 24
is_stateful = True
epochs = 50
dropout = 0.1
recurrent_dropout = 0
patience_coef = 0.1

test_size = batch_size
validation_size = batch_size

model_name = 'simple'

model_config = {
    'n_steps_in': n_steps_in,
    'n_steps_out': n_steps_out,
    'batch_size': batch_size,
    'is_stateful': is_stateful,
    'epochs': epochs,
    'dropout': dropout,
    'recurrent_dropout': recurrent_dropout,
    'patience_coef': patience_coef,
    'test_size': test_size,
    'validation_size': validation_size,
    'model_name': model_name
}

reversed_train, reversed_test, reversed_pred = process_lstm(datas, df, scaler, lambda_,
                                                            model_config)
analyze(m_out_file, reversed_train, reversed_test, reversed_pred, model_config, lambda_, df)
