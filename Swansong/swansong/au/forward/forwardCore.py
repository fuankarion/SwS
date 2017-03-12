import glob
import numpy as np
import os
from sklearn.metrics import classification_report
import sys

sys.path.insert(0, '/home/jcleon/Software/caffe/' + 'python')
import caffe #Caffe Lib
import caffe

import matlab.engine
import matlab
caffe.set_device(1)
caffe.set_mode_gpu()

#Globals
auArray = ['AU01', 'AU10', 'AU12', 'AU04', 'AU15', 'AU17', 'AU23', 'AU14', 'AU06', 'AU07']

def loadNetModel(auName, view, modelsRootPath):
    tagetModels = modelsRootPath + '/' + view + '/' + auName + '/*.caffemodel'
    print('tagetModels ', tagetModels)
    modelCand = glob.glob(tagetModels)
    print('Load model ', modelCand[0])
    net = caffe.Net('/home/jcleon/Storage/disk2/ModeslAFConcat2/deploy_Test.prototxt',
                    modelCand[0], caffe.TEST)
    return net

def createTransformers(net):
    # load input and configure preprocessing
    transformerRGB = caffe.io.Transformer({'dataRGB': net.blobs['dataRGB'].data.shape})

    #Config RGB transform
    transformerRGB.set_transpose('dataRGB', (2, 0, 1))  # move image channels to outermost dimension
    transformerRGB.set_mean('dataRGB', np.array([104.0, 117.0, 123.0]))             # subtract the dataset-mean value in each channel
    transformerRGB.set_raw_scale('dataRGB', 255.0)      # rescale from [0, 1] to [0, 255]
    transformerRGB.set_channel_swap('dataRGB', (2, 1, 0))  # swap channels from RGB to BGR

    # load input and configure preprocessing
    transformerFLOW = caffe.io.Transformer({'dataFLOW': net.blobs['dataFLOW'].data.shape})

    #Config Flow transform
    transformerFLOW.set_transpose('dataFLOW', (2, 0, 1))
    transformerFLOW.set_mean('dataFLOW', np.array([127.0, 128.0, 133.0]))
    transformerFLOW.set_raw_scale('dataFLOW', 255.0)
    transformerFLOW.set_channel_swap('dataFLOW', (2, 1, 0))
    
    return transformerFLOW, transformerRGB

def netForward(net, transformerFLOW, transformerRGB, imgRGB, imgFLow):
    
    #load images, ingnore if pair is not present
    try:
        im = caffe.io.load_image(imgRGB)
        imf = caffe.io.load_image(imgFLow)
    except:
        return net, False
    
    #Set flow and RGB data
    net.blobs['dataRGB'].reshape(1, 3, 224, 224)
    net.blobs['dataRGB'].data[...] = transformerRGB.preprocess('dataRGB', im)
    
    net.blobs['dataFLOW'].reshape(1, 3, 224, 224)
    net.blobs['dataFLOW'].data[...] = transformerFLOW.preprocess('dataFLOW', imf)
    net.forward()
    return net, True


def convertPaths(RGBPath, flowPath, trainFilesBase, baseJitterImagesPath):
    
    if '/home/afromero/Codes/FERA17/data/Faces/FERA17/BP4D/Train' in RGBPath:
        RGBPath = RGBPath.replace('/home/afromero/Codes/FERA17/data/Faces/FERA17/BP4D', trainFilesBase)
    elif '/home/afromero/Codes/FERA17/data/Faces/FERA17/BP4D/Jitter' in RGBPath:
        RGBPath = RGBPath.replace('/home/afromero/Codes/FERA17/data/Faces/FERA17/BP4D/Jitter', baseJitterImagesPath)
        jitterTokens = RGBPath.split('/')
        #print('Before ',flowPath)
        flowPath = flowPath.replace('/home/jcleon/Storage/disk2/resizedFera17-256Flow//Train', '/home/jcleon/Storage/disk2/Jitter/' + jitterTokens[6] + '/Train_Flow512')
        
    return RGBPath, flowPath
        
        
def getConvertedPathsFromGTLine(aLine, baseFlowImagesPath, baseRGBImagesPath, baseJitterImagesPath):
    lineTokens = aLine.split(' ')
    pathTokens = lineTokens[0].split('/')
    pathFLow = baseFlowImagesPath + '/' + pathTokens[-5] + '/' + pathTokens[-4] + '/' + pathTokens[-3] + '/' + pathTokens[-2] + '/' + pathTokens[-1]
    convertedRGBPath, convertedFlowPath = convertPaths(lineTokens[0], pathFLow, baseRGBImagesPath, baseJitterImagesPath)
    
    """
    print(' ')
    print('lineTokens[0] ', lineTokens[0])
    print('pathFLow', pathFLow)
    print('baseRGBImagesPath ', baseRGBImagesPath)
    
    print('convertedRGBPath ', convertedRGBPath)
    print('convertedFlowPath ', convertedFlowPath)
    print(' ')
    """
    
    return convertedRGBPath, convertedFlowPath 


def getTargetForward(aLine, targetForward):
    lineTokens = aLine.split(' ')
    pathTokens = lineTokens[0].split('/')
    
    print
    jitterToken = pathTokens[9]
    #print('targetForward ', targetForward)
    dirTargetForward = targetForward + '/' + pathTokens[-4] + '/' + pathTokens[-3] + '/' + pathTokens[-2]
    if not os.path.exists(dirTargetForward):
        os.makedirs(dirTargetForward)
        print ('created ', dirTargetForward)
    finalTargetForward = dirTargetForward + '/' + pathTokens[-1][:-4] + '.txt'
    #print('jitterToken ', jitterToken)
    if jitterToken == 'Jitter':
        finalTargetForward = dirTargetForward + '/' + pathTokens[-1][:-4] + '_' + pathTokens[10] + '.txt'
    
    return dirTargetForward, finalTargetForward

def getFileNameAndLabel(aLine):    
    tokensSpace = aLine.split(' ')
    label = tokensSpace[1]
    tokensName = tokensSpace[0].split('/')
    imgName = tokensName[-1]
    return imgName, label
    
def forwardFormGTFile(netParams, fileGT, targetForward, baseFlowImagesPath, layerData, 
                      au, baseRGBImagesPath, baseJitterImagesPath):
    gts = []
    preds = []
    scores = []
    
    net = netParams[0]
    transformerRGB = netParams[1]
    transformerFLOW = netParams[2]
      
    with open(fileGT) as f:
        content = f.readlines()

        idx = 0
        for aLine in content:
            convertedRGBPath, convertedFlowPath = getConvertedPathsFromGTLine(aLine, baseFlowImagesPath, baseRGBImagesPath, baseJitterImagesPath)
            net, flag = netForward(net, transformerFLOW, transformerRGB, convertedRGBPath, convertedFlowPath)

            if flag == False:       
                print('Pair not found for ', convertedRGBPath, convertedFlowPath)
                continue	
           
            dirTargetForward, finalTargetForward = getTargetForward(aLine, targetForward)
            targetLabels = dirTargetForward + '/labels.txt'
             
            """
            print('convertedRGBPath ', convertedRGBPath)
            print('convertedFlowPath ', convertedFlowPath)
            print('dirTargetForward ', dirTargetForward)
            print('finalTargetForward ', finalTargetForward)
            print('OK ')
            """

            with open(targetLabels, 'a') as gtFile:
                imgName, label = getFileNameAndLabel(aLine)
                tokensFTF = finalTargetForward.split('/')
                fileName = tokensFTF[-1][:-4]
                gtFile.write(fileName + '.jpg' + ',' + label)
                             
            gts.append(int(label))
            preds.append(net.blobs['softmax'].data[0].argmax())
            scores.append(net.blobs['softmax'].data[0][1])

            feat = net.blobs[layerData].data[0].flatten()#just in case
            
            
            with open(finalTargetForward, 'wb') as myFile:
                np.savetxt(myFile, feat, delimiter=",")

            if idx % 200 == 0 and idx > 0:
                print(classification_report(gts, preds))
               
            idx = idx + 1
         
    return gts, preds, scores

def forwardAUViewFold(au, view, fold, targetSet, trainFilesDir, baseTargetForward,
                      layerData, foldModelsPath, baseFlowImagesPath, baseRGBImagesPath, baseJitterImagesPath):
    auOhne0 = au.replace('0', '')
    
    fileGT = trainFilesDir + view + '/Training_' + auOhne0 + '.txt'
    modelsRootPath = foldModelsPath + '/fold_' + str(fold)
    targetForward = baseTargetForward + '/' + targetSet + '/' + au + '_' + view + '_Fold' + str(fold)
    
    print('fileGT ', fileGT)
    print('modelsRootPath ', modelsRootPath)
    print('targetForward ', targetForward)
    
    net = loadNetModel(au, view, modelsRootPath)
    transformerFLOW, transformerRGB = createTransformers(net)

    netParams = [net, transformerRGB, transformerFLOW]
    
    

    gts, preds, scores = forwardFormGTFile(netParams, fileGT, targetForward, baseFlowImagesPath, layerData,
                                           au, baseRGBImagesPath, baseJitterImagesPath)

    eng = matlab.engine.start_matlab()
    cs = classification_report(gts, preds)
    ps = eng.CalcRankPerformance(matlab.int8(gts), matlab.double(scores), 1, 'All')
    F1_MAX = max(np.array(ps['F1']))[0]

    with open(targetForward + '/overalReport.txt', 'a') as reportFile:
        reportFile.write(str(cs) + ' \n ' + ' F1Max: ' + str(F1_MAX))
        