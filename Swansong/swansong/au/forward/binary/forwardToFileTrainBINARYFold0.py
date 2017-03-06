import sys
sys.path.append('../')

from forwardCore import *

#exec
auName = 'AU01'
view = 'v1'
fileGT = '/home/jcleon/Storage/ssd0/fullFaceTrainFiles/' + view + '/Training.txt'
modelsRootPath = '/home/jcleon/fold/fold_0/'


basePathFlow='/home/jcleon/Storage/ssd0/Flow/Train'
targetForward = '/home/jcleon/Storage/ssd0/FeatsTrain/SoftMaxActivations/' + '/' + auName + '_' + view + '_Fold0'

net = loadNetModel(auName, view, modelsRootPath)
transformerFLOW, transformerRGB = createTransformers(net)

gts, preds, scores = forwardFormGTFile(net, transformerFLOW, transformerRGB, fileGT, targetForward,basePathFlow)

eng = matlab.engine.start_matlab()
cs = classification_report(gts, preds)
ps = eng.CalcRankPerformance(matlab.int8(gts), matlab.double(scores), 1, 'All')
F1_MAX = max(np.array(ps['F1']))[0]

with open(targetForward + '/overalReport.txt', 'a') as reportFile:
    reportFile.write(str(cs) + ' \n ' + ' F1Max: ' + str(F1_MAX))

