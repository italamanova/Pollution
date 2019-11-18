# from os import makedirs
from os import listdir
from os.path import isfile, isdir, join, exists
import json
from pathlib import Path

parent_dir_path = Path(__file__).parents[1]


def calcavg(path, params):
    print(listdir(path))
    files = [f for f in listdir(path) if isfile(join(path, f))]

    ar = []
    ti = []
    train_start = ''

    for i in range(0, len(files)):
        fullname = '%s/%s' % (path, files[i])
        with open(fullname) as json_file:
            data = json.load(json_file)
            tw = data.get('train_start')
            if i < 1:
                train_start = tw
            results = data.get('results')
            rmses = []
            times = data.get('time')
            model = data.get('model')
            itis = ''
            for n in range(0, len(params)):
                param = params[n]
                pv = model.get(param)
                itis += param + '=' + str(pv) + ','
            for obj in results:
                accuracy = obj.get('accuracy')
                rmse = float(accuracy.get('rmse'))
                rmses.append(rmse)
        allrmse = round(sum(rmses) / len(rmses), 3)

        ar.append(allrmse)
        ti.append(times)
        print('=%s:tw=%s, allrmse=%f, times=%f' % (fullname, tw, allrmse, times))
    avgrmse = round(sum(ar) / len(ar), 3)
    avgtime = round(sum(ti) / len(ti), 3)
    print('calcavg:avgrmse=%f, train_start=%s, avgtime=%f, itis=%s' % (avgrmse, train_start, avgtime, itis))
    return avgrmse, train_start, avgtime, itis


def calcall(pathes, params):
    print('pathes=', pathes)
    avgrmses = []
    train_starts = []
    fits = []
    its = []

    for path in pathes:
        for inner_dir in find_folders(path):
            inner_dir_path = join(path, inner_dir)
            avgrmse, train_start, avgtime, itis = calcavg(inner_dir_path, params)
            print('>%s:train_start=%s, avgrmse=%f, avgtime=%f, itis=%s' % (inner_dir_path, train_start, avgrmse, avgtime, itis))
            print('>======================================================')
            avgrmses.append(avgrmse)
            train_starts.append(train_start)
            fits.append(avgtime)
            its.append(itis)
    minrmse = min(avgrmses)
    j = 0
    for j in range(0, len(avgrmses)):
        if minrmse == avgrmses[j]:
            break
    best_train_start = train_starts[j]
    fit = fits[j]
    asis = its[j]
    print('callall:best_train_start=%s, minrmse=%f, fit=%f, asis=%s' % (best_train_start, minrmse, fit, asis))
    return minrmse, best_train_start, fit, asis

def find_folders(path):
    alldirs = []
    dirs = [f for f in listdir(path) if isdir(join(path, f))]
    for n in range(0, len(dirs)):
        alldirs.append(path + '/' + dirs[n])
    return alldirs



params = ['epochs', 'patience_coef']
# params = ['dropout', 'recurrent_dropout']
# params = ['validation_size']
# params = ['n_steps_in']
# params = ['batch_size', 'is_stateful']
# params = ['units_coef']
# params = ['train_window']
_folder_path = '%s/results/_simple/ep' % (parent_dir_path)
# dirs = finddirs([_folder_path])
dirs = [join(_folder_path, f) for f in find_folders(_folder_path)]
avgrmse, train_start, avgtime, itis = calcall(dirs, params)
# dirs2 = finddirs(dirs1)
# print(dirs2)
# avgrmse, train_start, avgtime, itis = calcall(dirs2, params)
# print('result:train_start=%s, avgrmse=%f, avgtime=%f, itis=%s' % (train_start, avgrmse, avgtime, itis))
