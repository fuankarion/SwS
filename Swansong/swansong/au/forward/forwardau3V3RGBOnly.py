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

caffe.set_device(1)
caffe.set_mode_gpu()

auIdx=3#Binary label on file
fileGT='/home/jcleon/Storage/ssd0/fullFaceTrainFiles/v3/Test.txt'

net = caffe.Net('/home/jcleon/Swansong/Swansong/swansong/au/deployRGBOnly.prototxt',
                '/home/jcleon/Storage/disk2/snapshot/v3/AU07/_iter_8596.caffemodel',
                caffe.TEST)


# load input and configure preprocessing
transformerRGB = caffe.io.Transformer({'dataRGB': net.blobs['dataRGB'].data.shape})

transformerRGB.set_transpose('dataRGB', (2,0,1))  # move image channels to outermost dimension
transformerRGB.set_mean('dataRGB', np.array([104.0, 117.0, 123.0]) )             # subtract the dataset-mean value in each channel
transformerRGB.set_raw_scale('dataRGB', 255.0)      # rescale from [0, 1] to [0, 255]
transformerRGB.set_channel_swap('dataRGB', (2,1,0))  # swap channels from RGB to BGR

def netForward(imgRGB):
    
    try:
        im = caffe.io.load_image(imgRGB)
    except:
	noresult={}
        noresult['loss']=np.array([[0.0,0.0]]) 
        print('Not found')
        print('imgRGB',imgRGB)
        return  noresult
    
    #note we can change the batch size on-the-fly
    #since we classify only one image, we change batch size from 10 to 1
    net.blobs['dataRGB'].reshape(1,3,224,224)
    net.blobs['dataRGB'].data[...] = transformerRGB.preprocess('dataRGB', im)

    return net.forward()

gts=[]
preds=[]

with open(fileGT) as f:
    content = f.readlines()
    
    idx=0
    for aLine in content:
        lineTokens=aLine.split(' ')     
        
        out=netForward(lineTokens[0])
	if out['loss'][0][0]==0.0 and out['loss'][0][0]==0.0:
            #img Not found
            continue		

        #print (out['loss'].argmax(),' ',lineTokens[4])
        
        gts.append(int(lineTokens[auIdx+1]))
        preds.append(out['loss'].argmax())

	if int(lineTokens[auIdx+1])==1:
	    print (out['loss'])
            print (lineTokens)   	   	
        
        idx=idx+1
        if idx%40==0:
            print('idx ',idx)
            rs=recall_score(gts, preds, average='binary',pos_label=1) 
            ps=precision_score(gts, preds, average='binary',pos_label=1)
            
            print('precision ',ps)
            print('recall ',rs)
            print(classification_report(gts, preds))

            #print('Accuracy ',acc)
