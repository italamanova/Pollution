from datetime import datetime
import json

from pmdarima import auto_arima
import pandas as pd

from helpers.converter import datetime_to_string
from helpers.visualizer import simple_plot
from myarima.arima import get_data
from helpers.accuracy import measure_accuracy


class AutoArimaSelection:
    def __init__(self, file, window_size, test_size,
                 start_p, max_p, start_q, max_q, max_d,
                 start_P, max_P, start_Q, max_Q, max_D,
                 m, information_criterion,
                 max_order=10,
                 d=None, D=None,
                 method=None, trend='c', solver='lbfgs',
                 suppress_warnings=True, error_action='warn', trace=True,
                 stepwise=True, seasonal=True, n_jobs=1,
                 out_file=None):
        self.df = get_data(file)
        self.window_size = window_size
        self.test_size = test_size

        self.start_p = start_p
        self.max_p = max_p

        self.start_q = start_q
        self.max_q = max_q
        self.d = d
        self.max_d = max_d

        self.start_P = start_P
        self.max_P = max_P
        self.start_Q = start_Q
        self.max_Q = max_Q
        self.D = D
        self.max_D = max_D
        self.max_order = max_order
        self.m = m

        self.information_criterion = information_criterion
        self.method = method
        self.trend = trend
        self.solver = solver
        self.suppress_warnings = suppress_warnings
        self.error_action = error_action
        self.trace = trace
        self.stepwise = stepwise
        self.seasonal = seasonal
        self.n_jobs = n_jobs

        self.out_file = out_file
        self.selection_result = []

    def __str__(self):
        pass

    def __repr__(self):
        pass

    def create_train_test_list(self):
        df_size = (len(self.df) // self.window_size) * self.window_size
        df = self.df.iloc[0:df_size].copy()
        train_size = self.window_size - self.test_size

        train_test_list = []
        for i in range(0, len(df), self.window_size):
            df_part = df.iloc[i:i + self.window_size].copy()
            train_test_list.append(
                (df_part.iloc[0:train_size].copy(), df_part.iloc[train_size:self.window_size].copy()))
        return train_test_list

    def print_to_file(self):
        f = open(self.out_file, "w")
        f.write(json.dumps(self.selection_result, default=datetime_to_string))
        f.close()

    def output_model(self, model):
        output = {}
        output['order'] = model.order
        output['seasonal_order'] = model.seasonal_order
        output['aic'] = model.aic()
        return output

    def build_model(self, train):
        stepwise_model = auto_arima(train, start_p=self.start_p, max_p=self.max_p,
                                    d=self.d, max_d=self.max_d,
                                    start_q=self.start_q, max_q=self.max_q,
                                    start_P=self.start_P, max_P=self.max_P,
                                    D=self.D, max_D=self.max_D,
                                    start_Q=self.start_Q, max_Q=self.max_P,
                                    seasonal=self.seasonal, m=self.m,
                                    max_order=self.max_order,
                                    trace=self.trace,
                                    error_action=self.error_action,
                                    suppress_warnings=self.suppress_warnings,
                                    stepwise=self.stepwise)

        return stepwise_model

    def predict_model(self, model, train, test):
        model.fit(train)
        future_forecast = model.predict(n_periods=len(test))
        metrics = measure_accuracy(test, future_forecast)
        return metrics

    def select_model(self):
        train_test_list = self.create_train_test_list()
        for train, test in train_test_list:

            output = {'train_start': train.index[0].strftime('%Y-%m-%d %H-%M-%S'),
                      'train_end': train.index[-1].strftime('%Y-%m-%d %H-%M-%S'),
                      'test_start': test.index[0].strftime('%Y-%m-%d %H-%M-%S'),
                      'test_end': test.index[-1].strftime('%Y-%m-%d %H-%M-%S')}
            print('BEFORE', output)
            print('BEFORE', len(train), len(test))
            model = self.build_model(train)
            output.update(self.output_model(model))
            metrics = self.predict_model(model, train, test)
            output.update(metrics)
            self.selection_result.append(output)
            print(output)
        self.print_to_file()
