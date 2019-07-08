from pathlib import Path

from myarima.arima_params_selection.arima_params_selection import ArimaParamsetersSelection
from myarima.arima_params_selection.selection_json_parser import get_max_orders

start_p = 0
max_p = 5
start_q = 0
max_q = 5
d = None
max_d = 2
start_P = 0
max_P = 3
start_Q = 0
max_Q = 3
D = None
max_D = 1
m = 24
seasonal = True
information_criterion = 'aic'
max_order = 10
stepwise = True
error_action = 'ignore'
out_file = 'arima_parameters_selection.json'

path_prepared = '%s/pollution_data/cut_data' % Path(__file__).parents[2]
path_out = '%s/pollution_data/arima_selection_results/parameters' % Path(__file__).parents[2]
file = '%s/Centar_PM25_prepared_1_year_arima_exp.csv' % path_prepared
# file = '%s/Centar_PM25_prepared.csv' % path_prepared

m_window_size = 1000
m_inner_window_size = 200

m_i_start = 0 * m_window_size
m_j_start = 0 * m_inner_window_size

m_out_file = '%s/%s_%s_%s_1_year.json' % (path_out, 'parameters', m_window_size, m_inner_window_size)

stand = ArimaParamsetersSelection(file, m_window_size, m_inner_window_size,
                                  start_p, max_p, start_q, max_q, max_d,
                                  start_P, max_P, start_Q, max_Q, max_D,
                                  m, information_criterion,
                                  max_order=max_order,
                                  d=d, D=D,
                                  error_action=error_action,
                                  stepwise=stepwise,
                                  seasonal=seasonal,
                                  out_file=m_out_file)

stand.select_model(i_start=m_i_start, j_start=m_j_start)

get_max_orders(m_out_file)
