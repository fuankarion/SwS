import matplotlib
import sys
matplotlib.use('Agg')
from pylab import *

from params import caffe_root
from params import models_root
sys.path.insert(0, caffe_root + 'python')
import caffe #Caffe Lib

from coreFunctions import trainNetworkBetterTrainDataLogTop2

caffe.set_device(0)
caffe.set_mode_gpu()

#Actual Params
solverPath = models_root+'/AUVGG16/solver.prototxt'
importModelPath = models_root+'/BaseModels/VGG16/VGG_ILSVRC_16_layers.caffemodel'


#Solver data 
solver = caffe.get_solver(solverPath)
solver.net.copy_from(importModelPath)


#Training Config (same as in solver)
batchesForTraining = 968
batchesUntillStep = 10;#about 8-10 is ok
maxSteps = 2;
niter = batchesForTraining * batchesUntillStep * maxSteps
test_iters = 146

#Log Data
targetLogFile = models_root + '/AUVGG16/logs/log.txt'
trainNetworkBetterTrainDataLogTop2(solver, niter, batchesForTraining, targetLogFile, test_iters, 5, 30)



