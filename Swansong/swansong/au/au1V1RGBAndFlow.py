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
solverPath = '/home/jcleon/Swansong/models/au/AU1V1VGG16/solver.prototxt'
#Log Data
targetLogFile ='/home/jcleon/Swansong/models/au/AU1V1VGG16/logs/log.txt'


#Solver data 
solver = caffe.get_solver(solverPath)
#Load andres Init
netObjects = caffe.Net('/home/jcleon/models/deployVGG16Binary.prototxt', 
                       '/home/jcleon/Storage/disk2/snapshot/v1/AU01/_iter_12348.caffemodel', caffe.TEST)

#copy andres weight into conv maps
solver.net.params['conv1_1'][0].data[...] = netObjects.params['conv1_1'][0].data 
solver.net.params['conv1_1'][1].data[...] = netObjects.params['conv1_1'][1].data

solver.net.params['conv1_2'][0].data[...] = netObjects.params['conv1_2'][0].data 
solver.net.params['conv1_2'][1].data[...] = netObjects.params['conv1_2'][1].data

solver.net.params['conv2_1'][0].data[...] = netObjects.params['conv2_1'][0].data
solver.net.params['conv2_1'][1].data[...] = netObjects.params['conv2_1'][1].data

solver.net.params['conv2_2'][0].data[...] = netObjects.params['conv2_2'][0].data
solver.net.params['conv2_2'][1].data[...] = netObjects.params['conv2_2'][1].data

solver.net.params['conv3_1'][0].data[...] = netObjects.params['conv3_1'][0].data
solver.net.params['conv3_1'][1].data[...] = netObjects.params['conv3_1'][1].data

solver.net.params['conv3_2'][0].data[...] = netObjects.params['conv3_2'][0].data
solver.net.params['conv3_2'][1].data[...] = netObjects.params['conv3_2'][1].data

solver.net.params['conv3_3'][0].data[...] = netObjects.params['conv3_3'][0].data
solver.net.params['conv3_3'][1].data[...] = netObjects.params['conv3_3'][1].data

solver.net.params['conv4_1'][0].data[...] = netObjects.params['conv4_1'][0].data
solver.net.params['conv4_1'][1].data[...] = netObjects.params['conv4_1'][1].data

solver.net.params['conv4_2'][0].data[...] = netObjects.params['conv4_2'][0].data
solver.net.params['conv4_2'][1].data[...] = netObjects.params['conv4_2'][1].data

solver.net.params['conv4_3'][0].data[...] = netObjects.params['conv4_3'][0].data
solver.net.params['conv4_3'][1].data[...] = netObjects.params['conv4_3'][1].data

solver.net.params['conv5_1'][0].data[...] = netObjects.params['conv5_1'][0].data
solver.net.params['conv5_1'][1].data[...] = netObjects.params['conv5_1'][1].data

solver.net.params['conv5_2'][0].data[...] = netObjects.params['conv5_2'][0].data
solver.net.params['conv5_2'][1].data[...] = netObjects.params['conv5_2'][1].data

solver.net.params['conv5_3'][0].data[...] = netObjects.params['conv5_3'][0].data
solver.net.params['conv5_3'][1].data[...] = netObjects.params['conv5_3'][1].data


solver.net.params['fc6'][0].data[...].flat = netObjects.params['fc6'][0].data.flat
solver.net.params['fc6'][1].data[...].flat = netObjects.params['fc6'][1].data.flat

solver.net.params['fc7'][0].data[...].flat = netObjects.params['fc7'][0].data.flat
solver.net.params['fc7'][1].data[...].flat = netObjects.params['fc7'][1].data.flat

#Training Config (same as in solver)
batchesForTraining = 1263
batchesUntillStep = 10;
maxSteps = 2;#Not quite there is a step
niter = batchesForTraining * batchesUntillStep * maxSteps
test_iters = 1432

trainNetworkBetterTrainDataLogTop2(solver, niter, batchesForTraining, targetLogFile, test_iters, 5, 30)



