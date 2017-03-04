import sys
import numpy as np
import lmdb
import glob

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

sys.path.insert(0, '/home/jcleon/Software/caffe/' + 'python')
import caffe #Caffe Lib
import caffe
from collections import defaultdict

import matlab.engine
import matlab
caffe.set_mode_gpu()

auIdx=3#Binary label on file
fileGT='/home/jcleon/Storage/ssd0/fullFaceTrainFiles/v3/Test.txt'

net = caffe.Net('/home/jcleon/Swansong/Swansong/swansong/au/deploy.prototxt',
                '/home/jcleon/Swansong/models/au/AU3V3VGG16/snapshots/_iter_12584.caffemodel',
                caffe.TEST)


# load input and configure preprocessing
transformerRGB = caffe.io.Transformer({'dataRGB': net.blobs['dataRGB'].data.shape})

transformerRGB.set_transpose('dataRGB', (2,0,1))  # move image channels to outermost dimension
transformerRGB.set_mean('dataRGB', np.array([104.0, 117.0, 123.0]) )             # subtract the dataset-mean value in each channel
transformerRGB.set_raw_scale('dataRGB', 255.0)      # rescale from [0, 1] to [0, 255]
transformerRGB.set_channel_swap('dataRGB', (2,1,0))  # swap channels from RGB to BGR


# load input and configure preprocessing
transformerFLOW = caffe.io.Transformer({'dataFLOW': net.blobs['dataFLOW'].data.shape})

transformerFLOW.set_transpose('dataFLOW', (2,0,1))
transformerFLOW.set_mean('dataFLOW', np.array([127.0, 128.0, 133.0])  )
transformerFLOW.set_raw_scale('dataFLOW', 255.0)
transformerFLOW.set_channel_swap('dataFLOW', (2,1,0))

def netForward(imgRGB,imgFLow):
    
    #load the image in the data layer
    #print('imgRGB',imgRGB)
    #print('imgFLow',imgFLow)
    try:
        im = caffe.io.load_image(imgRGB)
        imf=caffe.io.load_image(imgFLow)
    except:
	noresult={}
        noresult['loss']=np.array([[0.0,0.0]]) 
        print('Not found')
        print('imgRGB',imgRGB)
        print('imgFLow',imgFLow)
        return  noresult
    
    #note we can change the batch size on-the-fly
    #since we classify only one image, we change batch size from 10 to 1
    net.blobs['dataRGB'].reshape(1,3,224,224)
    net.blobs['dataRGB'].data[...] = transformerRGB.preprocess('dataRGB', im)

    #note we can change the batch size on-the-fly
    #since we classify only one image, we change batch size from 10 to 1
    net.blobs['dataFLOW'].reshape(1,3,224,224)
    net.blobs['dataFLOW'].data[...] = transformerFLOW.preprocess('dataFLOW', imf)
    return net.forward()

gts=[]
preds=[]
scores=[]

F1 = []; F1_MAX = []
eng = matlab.engine.start_matlab()
 
with open(fileGT) as f:
    content = f.readlines()
    
    idx=0
    for aLine in content:
        
        
        lineTokens=aLine.split(' ')
        pathTokens=lineTokens[0].split('/')
        
        #pathFLow='/home/jcleon/Storage/ssd1/FeraData/ValFLow/'+pathTokens[7]+'/'+pathTokens[8]+'/'+pathTokens[9]+'/'+pathTokens[10]
        pathFLow='/home/jcleon/Storage/ssd1/Flow/Val/'+pathTokens[7]+'/'+pathTokens[8]+'/'+pathTokens[9]+'/'+pathTokens[10]
        
        out=netForward(lineTokens[0],pathFLow)
        #print (out['loss'])
	if out['loss'][0][0]==0.0 and out['loss'][0][0]==0.0:
            #img Not found
            continue		
        #print(aLine)
        #print(out)
       
        #print (out['loss'].argmax(),' ',lineTokens[4])
        
        gts.append(int(lineTokens[auIdx+1]))
        preds.append(out['loss'].argmax())
        scores.append(out['loss'][0][1])
        

	if int(lineTokens[auIdx+1])==1:
	    print (out['loss'])
            print (lineTokens)   	 
            
       
        
        
        idx=idx+1
        if idx%50==0:
            print('idx ',idx)
            rs=recall_score(gts, preds, average='binary',pos_label=1) 
            ps=precision_score(gts, preds, average='binary',pos_label=1)
            
            print('precision ',ps)
            print('recall ',rs)
            print(classification_report(gts, preds))

            
            ps = eng.CalcRankPerformance(matlab.int8(gts), matlab.double(scores), 1, 'All')
            F1_MAX = max(np.array(ps['F1']))[0]
            print('F1_MAX ',F1_MAX)
            #print('Accuracy ',acc)
