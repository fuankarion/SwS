from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Sequential
import numpy as np
import os
import re

auArray = ['AU01', 'AU10', 'AU12', 'AU04', 'AU15', 'AU17', 'AU23', 'AU14', 'AU06', 'AU07']

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]	


def writeBulkLoadFeats(bulkLoadDir, aSubject, aTask, view, K, data, extra):
    bulkLoadFile = bulkLoadDir + '/' + aSubject + '-' + aTask + '-' + view + '-' + extra  + str(K)
    
    print('data.shape', data.shape)
    
    #with open(bulkLoadFile, 'wb') as blFile:
    #    np.savetxt(blFile, data, delimiter=",")
    np.save(bulkLoadFile, data)
   
        
        
def loadFeats(rootPath, aSubject, aTask, view, K, bulkLoadDir, extra):
    bulkLoadFile = bulkLoadDir + '/' + aSubject + '-' + aTask + '-' + view + '-' + extra  + str(K) + '.npy'
    if os.path.exists(bulkLoadFile):
        fromDisk = np.load(bulkLoadFile)
        print('fromBulkFileFeats.shape ', fromDisk.shape)
        return fromDisk, True
    
    print('No BulkFile Found at ', bulkLoadFile, ' read al txt')
    
    
    txtFilesPath = rootPath + '/' + aSubject + '/' + aTask + '/' + view + '/'
    
    allFilesInDir = os.listdir(txtFilesPath)
    allFilesInDir = sorted(allFilesInDir, key=natural_key)
    allFilesInDir = allFilesInDir[:-1]#pesky label file
    
    feats = np.zeros([0, 4096])#TODO ardcoded 4096
    
    idx = 0
    for aFile in allFilesInDir:
        tempPath = os.path.join(txtFilesPath, aFile)
        tempArray = np.loadtxt(tempPath)
        
        feats = np.insert(feats, idx, tempArray, axis=0)
        idx = idx + 1
        if idx % 100 == 0:
            print(idx, '/', len(allFilesInDir))
        
    return feats, False

def loadLabels(rootPath, aSubject, aTask, view, K, bulkLoadDir, extra):
    bulkLoadFile = bulkLoadDir + '/' + aSubject + '-' + aTask + '-' + view + '-' + extra  + str(K) + '.npy'
    if os.path.exists(bulkLoadFile):
        fromDisk = np.load(bulkLoadFile)
        print('fromBulkFileLabels.shape ', fromDisk.shape)
        return fromDisk, True
   
    print('No BulkFile Found at ', bulkLoadFile, ' read al txt')
    
    txtFilesPath = rootPath + '/' + aSubject + '/' + aTask + '/' + view + '/'
    
    labelFile = os.path.join(txtFilesPath, 'labels.txt')
    auIdx = auArray.index(au)
    labels = []
    with open(labelFile) as f:
        content = f.readlines()
        for aLine in content:
            lineTokens = aLine.split(',')
            fileName = lineTokens[0]
            label = lineTokens[auIdx + 1]
            
            labels.append(int(label))
            
    return np.array(labels), False
            
            
#map to Nsamples,K,4096
def reshapeTrainData(feats, labels, K):
    x = range(0, feats.shape[0])
    #print(x)
    reshaped = np.zeros([0, feats.shape[1] * K])
    labelsReshaped = np.zeros(0, dtype=int)
    for startIdx in range(0, len(x)-(K-1)):
        targetIndexes = x[startIdx:startIdx + K]      
        midOne = targetIndexes[len(targetIndexes) / 2]
        
        segment = feats[targetIndexes,:]
        labelSegment = labels[midOne]
        
        labelsReshaped = np.insert(labelsReshaped, startIdx, int(labelSegment), axis=0)
        reshaped = np.insert(reshaped, startIdx, segment.flatten(), axis=0)

    return reshaped, labelsReshaped

def loadSet(subjects, tasks):
    trainFeats = None
    trainLabels = None
    for aSubject in subjects:
        for aTask in tasks:
            print(' ')
            print('Subject ', aSubject)
            print('Task ', aTask)

            try:
                labels, bulkLoadLabels = loadLabels('/home/jcleon/Storage/ssd0/fc7Feats/AU01_v1_Model/', aSubject, aTask, view, K, bulkLoadDir, 'LabelsK')    
                feats, bulkLoadFeats = loadFeats('/home/jcleon/Storage/ssd0/fc7Feats/AU01_v1_Model/', aSubject, aTask, view, K, bulkLoadDir, 'FeatsK')
            except:
                print('Skip load ', aSubject, ' ', aTask)
                continue
            if not bulkLoadFeats:
                featsReshaped, labelsReshaped = reshapeTrainData(feats, labels, K)
                featsReshaped = np.reshape(featsReshaped, (featsReshaped.shape[0], 1, featsReshaped.shape[1]))#extra reshaoe for keras LSTM

                #Save to bulk load
                writeBulkLoadFeats(bulkLoadDir, aSubject, aTask, view, K, featsReshaped, 'FeatsK')
                writeBulkLoadFeats(bulkLoadDir, aSubject, aTask, view, K, labelsReshaped, 'LabelsK')
            else:
                featsReshaped = feats
                labelsReshaped = labels

            if trainFeats == None:
                trainFeats = np.copy(featsReshaped)
            else:
                trainFeats = np.concatenate((trainFeats, featsReshaped), axis=0)

            if trainLabels == None:
                trainLabels = np.copy(labelsReshaped)
            else:
                trainLabels = np.concatenate((trainLabels, labelsReshaped), axis=0)
            """    
            print('featsReshaped.shape', featsReshaped.shape)
            print('labelsReshaped.shape', labelsReshaped.shape)

            print('trainLabels.shape', trainLabels.shape)
            print('trainFeats.shape', trainFeats.shape)

    print('Final trainFeats.shape ', trainFeats.shape)
    print('Final trainLabels.shape ', trainLabels.hape)
    """
    return trainFeats, trainLabels

K = 5
au = 'AU01'
view = 'v1'

bulkLoadDir = '/home/jcleon/Storage/ssd0/bulkLoadFera'

trainSubjects = ['F007', 'F008', 'F009', 'F011', 'M001', 'M003', 'M004', 'M005', 'M006','rM001', 'rM002', 'rM003', 'rM004']
#trainSubjects = ['F011', 'F009','F011','F009']
#testSubjects = ['F011', 'F009']
testSubjects = ['M002', 'F010', 'rM007']

tasksTrain = ['T1', 'T5', 'T6', 'T7', 'T10', 'T11', 'T12', 'T13', 'T14']
tasksTest = ['T1', 'T5', 'T6', 'T7', 'T10', 'T11', 'T12', 'T13', 'T14']
#tasksTest = ['T1', 'T6', 'T10', 'T11', 'T14', 'T13']

trainFeats, trainLabels = loadSet(trainSubjects, tasksTrain)
testFeats, testLabels = loadSet(testSubjects, tasksTest)

print(trainFeats.shape)
print(trainLabels.shape)

print(testFeats.shape)
print(testLabels.shape)
""
model = Sequential()
model.add(Convolution1D(nb_filter=16, filter_length=1, border_mode='same', input_dim=K * 4096))#TODO care about hardcoded 4096
model.add(Activation('sigmoid'))

#model.add(Dense(64, activation='relu'))

model.add(LSTM(10))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy', 'fmeasure'])
print(model.summary())

classWeights = {0: 0.14, 1:0.86}#TODO!!
model.fit(trainFeats, trainLabels, nb_epoch=40, batch_size=300, verbose=1, class_weight=classWeights, validation_split=0.25)


print('XXXX Fitting done forward')
# Final evaluation of the model
scores = model.evaluate(trainFeats, trainLabels, verbose=1)
print(scores)
print(len(scores))

