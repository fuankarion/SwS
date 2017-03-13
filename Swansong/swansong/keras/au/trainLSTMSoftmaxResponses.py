from keras.callbacks import *
from trainCore import *


Ks = [2, 5, 10, 20, 30, 40, 50]
aus = ['AU01']
views = ['v2', 'v3', 'v4']
baseFeatsDir = '/home/jcleon/Storage/ssd0/featsDebug/Train'
trainFold = 0#Net used on forwad
evalFold = 0
modelTargetDir = '/home/jcleon/Storage/ssd0/ModelsLSTM'
reportTargetDir = '/home/jcleon/Storage/ssd0/ReportsLSTM'

for anAU in aus:
    for aView in views:
        for aK in Ks:
            jointModel = createModel(aK)
            trainedModel, trainLabelsUnbalanced, trainFeatsUnbalanced = trainLSTMModel(anAU, aView, trainFold, jointModel, aK, baseFeatsDir)
            evalReport = evalModel(anAU, aView, evalFold, trainedModel, trainLabelsUnbalanced, trainFeatsUnbalanced, aK, baseFeatsDir)
            jointModel.save(modelTargetDir + '/' + anAU + '-' + aView + '-K' + str(aK) + '.h5') 

            print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            print(evalReport)
            print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            
            f = open(reportTargetDir + '/' + anAU + '-' + aView + '-K' + str(aK) + '.txt', 'w')
            f.write(evalReport)
            f.close() 
            
            
