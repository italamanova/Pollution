import json
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler

from helpers.preparator import get_data_with_box_cox
from helpers.saver import print_to_file
from lstm.lstm_selection.analyzer import analyze, analyze_multiple
from lstm.lstm_selection.processor import process_lstm

path_prepared = '%s/data/centar' % Path(__file__).parents[2]
path_to_file_prepared = '%s/Centar_PM25_prepared.csv' % path_prepared

path_out = '%s/data/main_experiment/lstm_experiment_results' % Path(__file__).parents[2]
m_out_file = '%s/%s.json' % (path_out, 'multiple_lstm')

df, lambda_ = get_data_with_box_cox(path_to_file_prepared)
# df = get_data(path_to_file_prepared)
df = df.iloc[:480]
df_val = df[[df.columns[0]]].values.astype(float)

scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_df = scaler.fit_transform(df_val)
datas = [scaled_df, scaled_df]

model_configs = [{
    'n_steps_in': 24,
    'n_steps_out': 24,
    'batch_size': 24,
    'is_stateful': True,
    'epochs': 60,
    'dropout': 0.1,
    'recurrent_dropout': 0,
    'patience_coef': 0.1,
    'test_size': 24,
    'validation_size': 24,
    'model_name': 'simple'
},
    {
        'n_steps_in': 24,
        'n_steps_out': 24,
        'batch_size': 24,
        'is_stateful': True,
        'epochs': 50,
        'dropout': 0.1,
        'recurrent_dropout': 0,
        'patience_coef': 0.1,
        'test_size': 24,
        'validation_size': 24,
        'model_name': 'simple'
    }
]

print_to_file(m_out_file, [])

for config in model_configs:
    reversed_train, reversed_test, reversed_pred, train_df, test_df = process_lstm(datas, df, scaler, lambda_,
                                                                                   config)
    analyze_multiple(m_out_file, reversed_train, reversed_test, reversed_pred, config, lambda_, df)
