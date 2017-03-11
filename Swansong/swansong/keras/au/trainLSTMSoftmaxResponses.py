from keras.callbacks import *
from trainCore import *


Ks = [2, 5, 10, 20, 30, 40, 50]
aus = ['AU01']
views = ['v1']
baseFeatsDir = '/home/jcleon/Storage/disk2/foldFeats/Train'
trainFold = 0
evalFold = 1
modelTargetDir = '/home/jcleon/Storage/ssd0/ModelsLSTM'


for anAU in aus:
    for aView in views:
        for aK in Ks:
            jointModel = createModel(aK)
            trainedModel, trainLabelsUnbalanced, trainFeatsUnbalanced = trainLSTMModel(anAU, aView, trainFold, jointModel, aK,baseFeatsDir)
            evalModel(anAU, aView, evalFold, trainedModel, trainLabelsUnbalanced, trainFeatsUnbalanced, aK,baseFeatsDir)
            jointModel.save(modelTargetDir + '/' + anAU + '-' + aView + '-K' + str(aK) + '.h5') 
