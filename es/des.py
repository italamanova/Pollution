# from https://www.kaggle.com/andreicosma/time-series-double-exponential-smoothing
from pandas import Series
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from analysis.analyzer import get_data


def manual_es(file):
    t1 = time.time()
    csv_dataset = get_data(file)
    csv_dataset.plot()
    plt.show()

    optimal_gamma = None
    best_mse = None
    db = csv_dataset.values.astype('float32')
    # db = csv_dataset.iloc[:, :].values.astype('float32')
    # print(db)
    rmax = 99  # 9
    cof = 0.01  # 0.1
    mean_results_for_all_possible_alpha_gamma_values = np.zeros((rmax, rmax))
    for gamma in range(0, rmax):
        for alpha in range(0, rmax):
            pt = db[0]
            bt = db[1] - db[0]
            mean_for_alpha_gamma = np.zeros(len(db))
            mean_for_alpha_gamma[0] = np.power(db[0] - pt, 2)
            for i in range(0, len(db) - 2):
                temp_pt = ((alpha + 1) * cof) * db[i] + (1 - ((alpha + 1) * cof)) * (pt + bt)
                bt = ((gamma + 1) * cof) * (temp_pt - pt) + (1 - ((gamma + 1) * cof)) * bt
                pt = temp_pt
                mean_for_alpha_gamma[i] = np.power(db[i] - pt, 2)
            mean_results_for_all_possible_alpha_gamma_values[gamma][alpha] = np.mean(mean_for_alpha_gamma)
            optimal_gamma, optimal_alpha = np.unravel_index(
                np.argmin(mean_results_for_all_possible_alpha_gamma_values),
                np.shape(mean_results_for_all_possible_alpha_gamma_values))
    optimal_alpha = (optimal_alpha + 1) * cof
    optimal_gamma = (optimal_gamma + 1) * cof
    ##################
    best_mse = np.min(mean_results_for_all_possible_alpha_gamma_values)
    print("Best MSE = %s" % best_mse)
    print("Optimal alpha = %s" % optimal_alpha)
    print("Optimal gamma = %s" % optimal_gamma)
    # train
    pt = db[0]
    bt = db[1] - db[0]
    for i in range(1, len(db) - 2):
        temp_pt = optimal_alpha * db[i] + (1 - optimal_alpha) * (pt + bt)
        bt = optimal_gamma * (temp_pt - pt) + (1 - optimal_gamma) * bt
        pt = temp_pt
    print("P_t = %s" % pt)
    print("b_t = %s" % bt)
    t2 = time.time()
    # predict
    print('time use = %d' % (t2 - t1))
    print("Next observation = %s" % (pt + (1 * bt)))
    print("Real value = %f" % db[len(db) - 1])

    # forecast
    # forecast = np.zeros(len(db) + 1)
    # pt = db[0]
    # bt = db[1] - db[0]
    # forecast[0] = pt
    # for i in range(1, len(db)):
    #     temp_pt = optimal_alpha * db[i] + (1 - optimal_alpha) * (pt + bt)
    #     bt = optimal_gamma * (temp_pt - pt) + (1 - optimal_gamma) * bt
    #     pt = temp_pt
    #     forecast[i] = pt
    # forecast[-1] = pt + (1 * bt)
    # plt.plot(db,label = 'real data')
    # plt.plot(forecast, label = 'forecast')
    # plt.legend()
    # plt.show()
