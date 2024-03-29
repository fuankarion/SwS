import sys
sys.path.append('../')

from forwardCore import *

#exec
#aus = ['AU01', 'AU04']
#views = ['v1', 'v2']

aus = ['AU04', 'AU06']
views = ['v6']
fold = 1

trainFilesDir = '/home/jcleon/Storage/ssd0/fullFaceTrainFiles/'
foldModelsPath = '/home/jcleon/Storage/disk2/fold'
layerData = 'softmax'
baseTargetForward = '/home/jcleon/Storage/ssd0/featsDebug'

baseRGBImagesPath = '/home/jcleon/Storage/ssd0/RGB/'
baseFlowImagesPath = '/home/jcleon/Storage/disk2/resizedFera17-256Flow/'
baseJitterImagesPath = '/home/jcleon/Storage/disk2/Jitter'#Works also for flow

#allways forward with fold0 the train on fold1
for anAU in aus:
    for aView in views:
        forwardAUViewFold(anAU, aView, fold, 'Train', trainFilesDir, baseTargetForward, 
                          layerData, foldModelsPath, baseFlowImagesPath, baseRGBImagesPath, baseJitterImagesPath)
       
      
        
