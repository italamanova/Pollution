from datetime import datetime
import json

import numpy
from pmdarima import auto_arima
import pandas as pd

from helpers.converter import datetime_to_string
from helpers.visualizer import simple_plot
from myarima.arima import get_data
from helpers.accuracy import measure_accuracy


class ArimaParamsetersSelection:
    def __init__(self, file, window_size, inner_window_size,
                 start_p, max_p, start_q, max_q, max_d,
                 start_P, max_P, start_Q, max_Q, max_D,
                 m, information_criterion,
                 max_order=10,
                 d=None, D=None,
                 method=None, trend='c', solver='lbfgs',
                 suppress_warnings=True, error_action='warn', trace=True,
                 stepwise=False, seasonal=True, n_jobs=1,
                 out_file=None):
        self.df = get_data(file)
        self.df['number'] = numpy.arange(len(self.df))
        self.col_name = self.df.columns[0]
        self.window_size = window_size
        self.inner_window_size = inner_window_size

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

    def split_window(self, i_start, j_start):
        window_list = []
        df_size = (len(self.df) // self.window_size) * self.window_size
        df = self.df.iloc[0:df_size].copy()
        for i in range(i_start, len(df), self.window_size):
            inner_window_list = []
            for j in range(j_start, self.window_size, self.inner_window_size):
                df_inner_part = df.iloc[i:i + j + self.inner_window_size].copy()
                inner_window_list.append(df_inner_part)
            j_start = 0
            window_list.append(inner_window_list)
        return window_list

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
        _model = auto_arima(train, start_p=self.start_p, max_p=self.max_p,
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

        return _model

    def select_model(self, i_start, j_start):
        window_list = self.split_window(i_start=i_start, j_start=j_start)
        for window_number, outer_window in enumerate(window_list):
            output = {'window_start': outer_window[0].index[0].strftime('%Y-%m-%d %H-%M-%S'),
                      'window_end': outer_window[-1].index[-1].strftime('%Y-%m-%d %H-%M-%S'),
                      'num': window_number,
                      'inner': []}

            for inner_window in outer_window:
                inner_output = {}
                inner_output.update({'inner_window_start': inner_window.index[0].strftime('%Y-%m-%d %H-%M-%S'),
                                     'inner_window_end': inner_window.index[-1].strftime('%Y-%m-%d %H-%M-%S'),
                                     'range': '%s - %s' % (inner_window['number'][0], inner_window['number'][-1])
                                     })
                # print('BEFORE', output)
                # print('BEFORE', len(train), len(test))
                model = self.build_model(inner_window[self.col_name])
                inner_output.update(self.output_model(model))
                inner_output.update({'len': int(len(inner_window))})
                output['inner'].append(inner_output)
            self.selection_result.append(output)
            # print(self.selection_result)
            self.print_to_file()
