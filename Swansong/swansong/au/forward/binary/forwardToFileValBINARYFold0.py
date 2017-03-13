import sys
sys.path.append('../')

from forwardCoreNoJitter import *

#exec
aus = ['AU04', 'AU06']
views = ['v6']
fold = 0
layerData = 'softmax'
basePathFlow = '/home/jcleon/Storage/ssd0/Flow/Val'
targetForward = '/home/jcleon/Storage/ssd0/featsDebug/Val/'

for anAU in aus:
    for aView in views: 
        fileGT = '/home/jcleon/Storage/ssd0/fullFaceTrainFiles/' + aView + '/Test.txt'
        modelsRootPath = '/home/jcleon/Storage/disk2/fold/fold_' + str(fold)+'/'
        targetForward = targetForward + '/' + anAU + '_' + aView + '_Fold' + str(fold)

        net = loadNetModel(anAU, aView, modelsRootPath)
        transformerFLOW, transformerRGB = createTransformers(net)

        gts, preds, scores = forwardFormGTFile(net, transformerFLOW, transformerRGB, fileGT, targetForward, basePathFlow, layerData, anAU)

        eng = matlab.engine.start_matlab()
        cs = classification_report(gts, preds)
        ps = eng.CalcRankPerformance(matlab.int8(gts), matlab.double(scores), 1, 'All')
        F1_MAX = max(np.array(ps['F1']))[0]

        with open(targetForward + '/overalReport.txt', 'a') as reportFile:
            reportFile.write(str(cs) + ' \n ' + ' F1Max: ' + str(F1_MAX))

