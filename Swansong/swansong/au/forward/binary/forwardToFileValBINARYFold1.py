import sys
sys.path.append('../')

from forwardCoreNoJitter import *

aus = ['AU04']
views = ['v2']
fold=0

for anAU in aus:
    for aView in views:
        #exec
        auName = anAU
        view =aView
        fileGT = '/home/jcleon/Storage/ssd0/fullFaceTrainFiles/' + view + '/Test.txt'
        modelsRootPath = '/home/jcleon/Storage/disk2/fold/fold_1/'
        layerData = 'softmax'

        basePathFlow = '/home/jcleon/Storage/ssd0/Flow/Val'
        targetForward = '/home/jcleon/Storage/disk2/foldFeats/Val/' + '/' + auName + '_' + view + '_Fold1'

        net = loadNetModel(auName, view, modelsRootPath)
        transformerFLOW, transformerRGB = createTransformers(net)

        gts, preds, scores = forwardFormGTFile(net, transformerFLOW, transformerRGB, fileGT, targetForward, basePathFlow, layerData, auName)

        eng = matlab.engine.start_matlab()
        cs = classification_report(gts, preds)
        ps = eng.CalcRankPerformance(matlab.int8(gts), matlab.double(scores), 1, 'All')
        F1_MAX = max(np.array(ps['F1']))[0]

        with open(targetForward + '/overalReport.txt', 'a') as reportFile:
            reportFile.write(str(cs) + ' \n ' + ' F1Max: ' + str(F1_MAX))

