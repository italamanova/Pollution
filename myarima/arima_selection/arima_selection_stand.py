from pathlib import Path

from myarima.arima_selection.arima_selection import AutoArimaSelection

window_size = 24 * 7
test_size = 12
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
information_criterion = 'aic'
max_order = 10
stepwise = False
error_action = 'ignore'
out_file = 'arima_select.json'

path_prepared = '%s/data/cut_data' % Path(__file__).parents[2]
path_out = '%s/data/arima_selection_parameter_results/1' % Path(__file__).parents[2]
# file = '%s/Centar_PM25_prepared_1year.csv' % path_prepared
file = '%s/Centar_PM25_prepared_1year.csv' % path_prepared

windows = [24*5, 24 * 7, 24 * 14, 24 * 30]
test_sizes = [4, 12, 24, 24]

for i in range(0, len(windows)):
    m_window_size = windows[i]
    m_test_size = test_sizes[i]
    print(m_window_size, m_test_size)
    m_out_file = '%s/%s_%s_%s.json' % (path_out, i, m_window_size, m_test_size)

    stand = AutoArimaSelection(file, m_window_size, m_test_size,
                               start_p, max_p, start_q, max_q, max_d,
                               start_P, max_P, start_Q, max_Q, max_D,
                               m, information_criterion,
                               max_order=max_order,
                               d=d, D=D,
                               error_action=error_action,
                               out_file=m_out_file)

    stand.select_model()
