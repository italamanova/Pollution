import random
from pathlib import Path

import numpy
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from helpers.decorators import timeit
from helpers.preparator import get_data_with_box_cox, reverse_box_cox_for_lstm
from lstm.lstm_selection.predictor import Predictor
from lstm.lstm_selection.preparator import prepare_data, inverse_scale


@timeit
def process_lstm(datas, df, scaler, lambda_, model_config):
    n_steps_in = model_config['n_steps_in']
    n_steps_out = model_config['n_steps_out']
    sum_steps = n_steps_in + n_steps_out
    batch_size = model_config['batch_size']
    is_stateful = model_config['is_stateful']
    epochs = model_config['epochs']
    dropout = model_config['dropout']
    recurrent_dropout = model_config['recurrent_dropout']
    patience_coef = model_config['patience_coef']
    test_size = model_config['test_size']
    validation_size = model_config['validation_size']
    model_name = model_config['model_name']

    X, y, Xv, yv, Xt, yt = prepare_data(datas, validation_size, test_size, batch_size, n_steps_in, n_steps_out)

    predictor = Predictor(X, y, Xv, yv, Xt, yt, n_steps_in, n_steps_out, batch_size, is_stateful,
                          epochs=epochs,
                          dropout=dropout,
                          recurrent_dropout=recurrent_dropout,
                          patience_coef=patience_coef)
    yhat, history = predictor.predict(model_name=model_name)

    inverse_test, inverse_pred = inverse_scale(scaler, Xt, yhat)

    reversed_train, reversed_test, reversed_pred = reverse_box_cox_for_lstm(lambda_,
                                                                            inverse_test, inverse_pred,
                                                                            df, sum_steps)
    train_df = df[:len(df) - sum_steps]
    test_df = df[len(df) - sum_steps:]

    return reversed_train, reversed_test, reversed_pred, train_df, test_df
