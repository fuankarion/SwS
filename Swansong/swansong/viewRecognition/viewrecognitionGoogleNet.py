import matplotlib
import sys
matplotlib.use('Agg')
from pylab import *

from params import caffe_root
from params import models_root
sys.path.insert(0, caffe_root + 'python')
import caffe #Caffe Lib

from coreFunctions import trainNetworkBetterTrainDataLogTop2

#Only gpu 0
caffe.set_device(0)
caffe.set_mode_gpu()

#Actual Params
solverPath = models_root+'/ViewsGoogleNet/solver.prototxt'
importModelPath = models_root+'/BaseModels/GoogleNet/imagenet_googlenet.caffemodel'

#Solver data 
solver = caffe.get_solver(solverPath)
solver.net.copy_from(importModelPath)


#Training Config (same as in solver)
batchesForTraining = 6610
batchesUntillStep = 1;#about 8-10 is ok
maxSteps = 2;
niter = batchesForTraining * batchesUntillStep * maxSteps
test_iters = 10647

#Log Data
targetLogFile = models_root + '/ViewsGoogleNet/logs/log.txt'
trainNetworkBetterTrainDataLogTop2(solver, niter, batchesForTraining, targetLogFile, test_iters, 8, 10)


