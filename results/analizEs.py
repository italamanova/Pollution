
from os import listdir
from os.path import isfile, isdir, join
import json
def eachsavg(eachs, steps):
	next = eachs[0 : steps + 1]
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
		#print(files[i])
		# one file
		acc2 = []
		time2 = []
		fullname = '%s/%s' %(path, files[i])
#################### start file
		with open(fullname) as json_file:
			data = json.load(json_file)
			tw = int(data.get('train_window'))
			if i < 1:
				train_window = tw
				#print(tw)
			results = data.get('results')
			acc1, time1 = calcresult(results, steps)
			avgtime1 = round(sum(time1)/len(time1), 3)
			avgacc1 = round(sum(acc1)/len(acc1), 3)
			acc2.append(avgacc1)
			time2.append(avgtime1)
#################### end file
		if len(acc2) > 0:
			avgtime2 = round(sum(time2)/len(time2), 3)
			avgacc2 = round(sum(acc2)/len(acc2), 3)
			acc3.append(avgacc2)
			time3.append(avgtime2)
	avgacc3 = 1000
	avgtime3 = 1000
	if len(acc3) > 0:
		avgacc3 = round(sum(acc3)/len(acc3), 3)
		avgtime3 = round(sum(time3)/len(time3), 3)
	return avgacc3, train_window, avgtime3 

def finddirs(path):
	alldirs = []
	for k in range(0, len(path)):
		dirs = [f for f in listdir(path[k]) if isdir(join(path[k], f))]
		for n in range(0, len(dirs)):
			alldirs.append(path[k]+'/'+dirs[n])
	return alldirs

#pathes = ['./_es/120', './_es/240', './_es/360', './_es/480', './_es/720', './_es/960'
#	, './_es/24', './_es/48', './_es/72', './_es/96', './_es/120', './_es/144', './_es/168', './_es/196', './_es/220'
#	]
path = ['_arima']
pathes = finddirs(path)
#print(pathes)
stepstest = [-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]  
for steps in stepstest:
	accs = []
	train_windows = []
	fits = []
	for path in pathes: 
		#print('steps=%d' % steps)
		acc_, train_window_, time_ = calcavg(path, steps)
		#print('train_window=%d, avgrmse=%f, avgtime=%f' % (train_window, avgrmse, avgtime))	
		accs.append(acc_)
		train_windows.append(train_window_)
		fits.append(time_)
	#print(accs)
	accmin = min(accs)
	j = 0
	for j in range(0, len(accs)):
		if accmin == accs[j]:
			break
	best_train_window = train_windows[j]
	fit = fits[j]
	print('steps=%d, best_train_window=%d, accmin=%f, fit=%f' % (steps, best_train_window, accmin, fit))

	