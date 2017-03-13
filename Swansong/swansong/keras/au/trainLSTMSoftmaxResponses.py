from keras.callbacks import *
from trainCore import *


Ks = [2,5,10,20,30,40,50]
aus = ['AU04']
views = ['v2']
baseFeatsDir = '/home/jcleon/Storage/disk2/foldFeats/Train'
trainFold =1 
evalFold = 1
modelTargetDir = '/home/jcleon/Storage/ssd0/ModelsLSTM'
reportTargetDir='/home/jcleon/Storage/ssd0/ReportsLSTM'

for anAU in aus:
    for aView in views:
        for aK in Ks:
            jointModel = createModel(aK)
            trainedModel, trainLabelsUnbalanced, trainFeatsUnbalanced = trainLSTMModel(anAU, aView, trainFold, jointModel, aK,baseFeatsDir)
            evalReport=evalModel(anAU, aView, evalFold, trainedModel, trainLabelsUnbalanced, trainFeatsUnbalanced, aK,baseFeatsDir)
            jointModel.save(modelTargetDir + '/' + anAU + '-' + aView + '-K' + str(aK) + '.h5') 

            print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            print(evalReport)
            print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            
            f = open(reportTargetDir+'/'+ anAU + '-' + aView + '-K' + str(aK)+'.txt', 'w')
            f.write(evalReport)
            f.close() 
            
            
