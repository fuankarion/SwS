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


def loadNetModel(auName, view, modelsRootPath):
    #TODO this is a param
    tagetModels = modelsRootPath + view + '/' + auName + '/*.caffemodel'
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


def forwardFormGTFile(net, transformerFLOW, transformerRGB, fileGT, targetForward, basePathFLow):
    gts = []
    preds = []
    scores = []
      
    with open(fileGT) as f:
        content = f.readlines()

        idx = 0
        for aLine in content:
            lineTokens = aLine.split(' ')
            pathTokens = lineTokens[0].split('/')

            pathFLow = basePathFLow + '/' + pathTokens[7] + '/' + pathTokens[8] + '/' + pathTokens[9] + '/' + pathTokens[10]

            #print('Path RGB ', lineTokens[0])
            #print('Path Flow ', pathFLow)
            net, flag = netForward(net, transformerFLOW, transformerRGB, lineTokens[0], pathFLow)

            #print (out['loss'])
            if flag == False:
                print('Pair not found for ', lineTokens[0])
                continue	

            tempTargetForward = targetForward + '/' + pathTokens[7] + '/' + pathTokens[8] + '/' + pathTokens[9]
            if not os.path.exists(tempTargetForward):
                os.makedirs(tempTargetForward)
                print ('created ', tempTargetForward)

            finalTargetForward = tempTargetForward + '/' + pathTokens[10][:-4] + '.txt'
            targetLabels = tempTargetForward + '/labels.txt'

            with open(targetLabels, 'a') as gtFile:
                gtFile.write(pathTokens[10] + ',' + lineTokens[1] + ',' + lineTokens[2] 
                             + ',' + lineTokens[3] + ',' + lineTokens[4] + ',' + lineTokens[5] 
                             + ',' + lineTokens[6] + ',' + lineTokens[7] + ',' + lineTokens[8] 
                             + ',' + lineTokens[9] + ',' + lineTokens[10])
                             
                             
            gts.append(int(lineTokens[1]))
            preds.append(net.blobs['softmax'].data[0].argmax())
            scores.append(net.blobs['softmax'].data[0][1])

            softMaxFeat = net.blobs['softmax'].data[0].flatten()#just in case
            with open(finalTargetForward, 'wb') as myFile:
                np.savetxt(myFile, softMaxFeat, delimiter=",")

            if idx % 200 == 0:
                print('Forwards ', idx)
                print(classification_report(gts, preds))
               
            idx = idx + 1
         
    return gts, preds, scores
