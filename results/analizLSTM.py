#from os import makedirs
from os import listdir
from os.path import isfile, isdir, join, exists
import json
def calcavg(path, params):
	files = [f for f in listdir(path) if isfile(join(path, f))]
	#print(files)
	ar = []
	ti = []
	train_start = ''
	for i in range(0, len(files)):
		#print(files[i])
		fullname = '%s/%s' %(path, files[i])
		#print(fullname)
		with open(fullname) as json_file:
			data = json.load(json_file)
			tw = data.get('train_start')
			if i < 1:
				train_start = tw
				#print(tw)
			results = data.get('results')
			rmses = []
			#times = []
			times = data.get('time')
			model = data.get('model')
			itis = ''
			for n in range(0, len(params)):
				param = params[n]
				pv = model.get(param)
				itis += param+'='+str(pv)+','
			for obj in results:
				accuracy = obj.get('accuracy')
				#time = obj.get('time')
				rmse = float(accuracy.get('rmse')) 
				#print('time=%f' % time)
				#times.append(time)
				#print(rmse)
				rmses.append(rmse)
				#eacc = obj.get('each_sample_accuracy')
				#print(eacc)
				#print('detail:tw=%s, rmse=%f,times=%f' % (tw, rmse, times))
		#alltimes = round(sum(times)/len(times), 3)
		allrmse = round(sum(rmses)/len(rmses), 3)
		
		#print('allrmse=%d' % allrmse)
		ar.append(allrmse)
		ti.append(times)
		print('=%s:tw=%s, allrmse=%f, times=%f' % (fullname, tw, allrmse, times))
	#print(ar)
	avgrmse = round(sum(ar)/len(ar), 3)
	avgtime = round(sum(ti)/len(ti), 3)
	#print('train_window=%d, avgrmse=%f' % (train_window, avgrmse))	
	print('calcavg:avgrmse=%f, train_start=%s, avgtime=%f, itis=%s' % (avgrmse, train_start, avgtime, itis))
	return avgrmse, train_start, avgtime, itis 

def calcall(pathes, params):
	print('pathes=', pathes)
	avgrmses = []
	train_starts = []
	fits = []
	its = []

	for path in pathes: 
		avgrmse, train_start, avgtime, itis = calcavg(path, params)
		print('>%s:train_start=%s, avgrmse=%f, avgtime=%f, itis=%s' % (path, train_start, avgrmse, avgtime, itis))	
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
#pathes = ['./_lstm/0', './_lstm/1000']#	, './_lstm/2000'	, './_lstm/3000'	, './_lstm/4000'	, './_lstm/5000'	, './_lstm/6000'	, './_lstm/7000'	, './_lstm/8000'

def finddirs(path):
	alldirs = []
	for k in range(0, len(path)):
		dirs = [f for f in listdir(path[k]) if isdir(join(path[k], f))]
		for n in range(0, len(dirs)):
			alldirs.append(path[k]+'/'+dirs[n])
	return alldirs
#params = ['epochs', 'patience_coef']
#params = ['dropout', 'recurrent_dropout']
#params = ['validation_size']
#params = ['n_steps_in']
#params = ['batch_size', 'is_stateful']
#params = ['units_coef']
params = ['train_window']
path = ['_simple/twless']
dirs1 = finddirs(path)
print(dirs1)
avgrmse, train_start, avgtime, itis = calcall(dirs1, params)
#dirs2 = finddirs(dirs1)
#print(dirs2)
#avgrmse, train_start, avgtime, itis = calcall(dirs2, params)
#print('result:train_start=%s, avgrmse=%f, avgtime=%f, itis=%s' % (train_start, avgrmse, avgtime, itis))	
	