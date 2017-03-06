from forwardCore import *

auName = 'AU01'
view = 'v1'
fileGT = '/home/jcleon/Storage/ssd0/fullFaceTrainFiles/' + view + '/Test.txt'
targetForward = '/home/jcleon/Storage/ssd0/softMaxResponses/' + '/' + auName + '_' + view + '_FullTrain'
modelsRootPath = '/home/jcleon/Storage/disk2/ModeslAFConcat2/snapshot_concat2/'

net = loadNetModel(auName, view, modelsRootPath)
transformerFLOW, transformerRGB = createTransformers(net)

gts, preds, scores = forwardFormGTFile(net, transformerFLOW, transformerRGB, fileGT, targetForward)

eng = matlab.engine.start_matlab()
cs = classification_report(gts, preds)
ps = eng.CalcRankPerformance(matlab.int8(gts), matlab.double(scores), 1, 'All')
F1_MAX = max(np.array(ps['F1']))[0]

with open(targetForward + '/overalReport.txt', 'a') as reportFile:
    reportFile.write(str(cs) + ' \n ' + ' F1Max: ' + str(F1_MAX))

