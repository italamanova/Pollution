import json
import os

from os import listdir
from os.path import isfile, isdir, join
import json
from pathlib import Path

from helpers.plotter import plot_heatmap

parent_dir_path = Path(__file__).parents[1]


def eachsavg(eachs, steps):
    next = eachs[0: steps + 1]
    avg = sum(next) / len(next)
    return avg


def calcresult(results, steps):
    acc1 = []
    time1 = []
    for obj in results:
        accuracy = obj.get('accuracy')
        rmse = float(accuracy.get('rmse'))
        each_sample_accuracy = obj.get('each_sample_accuracy')
        time = obj.get('time')
        acc = rmse
        if steps >= 0:
            acc = eachsavg(each_sample_accuracy, steps)
        acc1.append(acc)
        time1.append(time)
    return acc1, time1


def calcavg(path, steps):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    # files in dir
    acc3 = []
    time3 = []
    train_window = 0
    ################### start files in dir
    for i in range(0, len(files)):
        # print(files[i])
        # one file
        acc2 = []
        time2 = []
        fullname = '%s/%s' % (path, files[i])
        #################### start file
        with open(fullname) as json_file:
            data = json.load(json_file)
            tw = int(data.get('train_window'))
            if i < 1:
                train_window = tw
            # print(tw)
            results = data.get('results')
            acc1, time1 = calcresult(results, steps)
            avgtime1 = round(sum(time1) / len(time1), 3)
            avgacc1 = round(sum(acc1) / len(acc1), 3)
            acc2.append(avgacc1)
            time2.append(avgtime1)
        #################### end file
        if len(acc2) > 0:
            avgtime2 = round(sum(time2) / len(time2), 3)
            avgacc2 = round(sum(acc2) / len(acc2), 3)
            acc3.append(avgacc2)
            time3.append(avgtime2)
    avgacc3 = 1000
    avgtime3 = 1000
    if len(acc3) > 0:
        avgacc3 = round(sum(acc3) / len(acc3), 3)
        avgtime3 = round(sum(time3) / len(time3), 3)
    return avgacc3, train_window, avgtime3


def get_inner_folders(path):
    folders = []
    for root, directory, files in os.walk(path):
        for folder in directory:
            folders.append(os.path.join(root, folder))
    return folders


def check_files(folder_path):
    inner_folders = get_inner_folders(folder_path)
    _steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    _result_trains = []
    _result_accuracy = []
    for step in _steps:
        accuracy = []
        train_windows = []
        for folder in inner_folders:
            acc_, train_window_, time_ = calcavg(folder, step)
            accuracy.append(acc_)
            train_windows.append('koko %s' %train_window_)
        _result_trains = train_windows.copy()
        _result_accuracy.append(accuracy)

    return _steps, _result_trains, _result_accuracy


_folder_path = '%s/results/_arima' % (parent_dir_path)
# print(_folder_path)
steps, result_trains, result_accuracy = check_files(_folder_path)
print('STEPS', steps)
print('TRAIN', result_trains)
print('ACC', result_accuracy)
plot_heatmap(x=result_trains, y=steps, z=result_accuracy)
