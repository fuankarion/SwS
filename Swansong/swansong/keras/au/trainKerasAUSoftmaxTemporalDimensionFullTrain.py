from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers.convolutional import MaxPooling1D
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.optimizers import SGD
import numpy as np
import os
import re

auArray = ['AU01', 'AU10', 'AU12', 'AU04', 'AU15', 'AU17', 'AU23', 'AU14', 'AU06', 'AU07']

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]	


def writeBulkLoadFeats(bulkLoadDir, aSubject, aTask, view, K, data, extra):
    bulkLoadFile = bulkLoadDir + '/' + aSubject + '-' + aTask + '-' + view + '-' + extra  + str(K)
    np.save(bulkLoadFile, data)
        
def testBulkLoadData(bulkLoadFile):
    if os.path.exists(bulkLoadFile):
        fromDisk = np.load(bulkLoadFile)
        print('fromBulkFileFeats.shape ', fromDisk.shape)
        return fromDisk, True
    
    else:
        print('No Feat BulkFile Found at ', bulkLoadFile, ' read al txt')
        return None, False
   
def loadFeats(rootPath, aSubject, aTask, view, K, bulkLoadDir, extra):
    bulkLoadFile = bulkLoadDir + '/' + aSubject + '-' + aTask + '-' + view + '-' + extra  + str(K) + '.npy'
    data, flag = testBulkLoadData(bulkLoadFile)
    if flag:
        return data, flag
    
    txtFilesPath = rootPath + '/' + aSubject + '/' + aTask + '/' + view + '/'
    print('txtFilesPath ', txtFilesPath)
    
    allFilesInDir = os.listdir(txtFilesPath)
    allFilesInDir = sorted(allFilesInDir, key=natural_key)
    allFilesInDir = allFilesInDir[:-1]#pesky label file
    
    allfeats = np.zeros([len(allFilesInDir), 4096])#TODO hardcoded 4096
    
    idx = 0
    for aFile in allFilesInDir:
        tempPath = os.path.join(txtFilesPath, aFile)
        tempFeats = np.loadtxt(tempPath)
        
        allfeats[idx,:] = tempFeats
        idx = idx + 1
        if idx % 100 == 0:
            print(idx, '/', len(allFilesInDir))
        
    return allfeats, False

def loadLabels(rootPath, aSubject, aTask, view, K, bulkLoadDir, extra):
    bulkLoadFile = bulkLoadDir + '/' + aSubject + '-' + aTask + '-' + view + '-' + extra  + str(K) + '.npy'
    data, flag = testBulkLoadData(bulkLoadFile)
    if flag:
        return data, flag
    
    
    txtFilesPath = rootPath + '/' + aSubject + '/' + aTask + '/' + view + '/'
    print('txtFilesPath ', txtFilesPath)
    
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
            
            
#map to ?,K,4096
def reshapeTrainData(feats, labels, K):
    print('Reshape')
    x = range(0, feats.shape[0])

    reshaped = np.zeros([len(x)-(K-1), K, feats.shape[1]])
    labelsReshaped = np.zeros(0, dtype=int)
    
    for startIdx in range(0, len(x)-(K-1)):
        targetIndexes = x[startIdx:startIdx + K]  
        midOne = targetIndexes[len(targetIndexes) / 2]
        
        #Feats
        segment = feats[targetIndexes,:]
        reshaped[startIdx,:,:] = segment
        
        #Labels
        labelSegment = labels[midOne]
        labelsReshaped = np.insert(labelsReshaped, startIdx, int(labelSegment), axis=0)

    return reshaped, labelsReshaped

def loadSet(subjects, tasks, txtDir):
    trainFeats = None
    trainLabels = None
    for aSubject in subjects:
        for aTask in tasks:
            print(' ')
            print('Subject ', aSubject)
            print('Task ', aTask)

            try:
                labels, bulkLoadLabels = loadLabels(txtDir, aSubject, aTask, view, K, bulkLoadDir, 'LabelsK')    
                feats, bulkLoadFeats = loadFeats(txtDir, aSubject, aTask, view, K, bulkLoadDir, 'FeatsK')
            except Exception  as err:
                print(err)
                print('Skip load ', aSubject, ' ', aTask)
                continue
            
            if not bulkLoadFeats:
                featsReshaped, labelsReshaped = reshapeTrainData(feats, labels, K)

                #Save to bulk load
                writeBulkLoadFeats(bulkLoadDir, aSubject, aTask, view, K, featsReshaped, 'FeatsK')
                writeBulkLoadFeats(bulkLoadDir, aSubject, aTask, view, K, labelsReshaped, 'LabelsK')
            else:
                featsReshaped = feats
                labelsReshaped = labels

            #Append to current Feats
            if trainFeats == None:
                trainFeats = np.copy(featsReshaped)
            else:
                trainFeats = np.concatenate((trainFeats, featsReshaped), axis=0)

            if trainLabels == None:
                trainLabels = np.copy(labelsReshaped)
            else:
                trainLabels = np.concatenate((trainLabels, labelsReshaped), axis=0)
                
            print('featsReshaped.shape', featsReshaped.shape)
            print('labelsReshaped.shape', labelsReshaped.shape)

            print('trainLabels.shape', trainLabels.shape)
            print('trainFeats.shape', trainFeats.shape)
    """        
    print('Final trainFeats.shape ', trainFeats.shape)
    print('Final trainLabels.shape ', trainLabels.hape)
    """
    return trainFeats, trainLabels

K = 5
au = 'AU01'
view = 'v1'

txtFeatsDir = '/home/jcleon/Storage/ssd0/softMaxResponses/AU01_v1_FullTrain'
bulkLoadDir = '/home/jcleon/Storage/ssd0/bulkLoadFeraAU01_v1_FullTrainTemporalDimension'

trainSubjects = ['F007', 'F009', 'M001', 'M003', 'rM001', 'rM001']
testSubjects = ['F008', 'M002', 'rM002']

tasksTrain = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'T13', 'T14']
tasksTest = ['T1', 'T11', 'T12', ]

trainFeats, trainLabels = loadSet(trainSubjects, tasksTrain, txtFeatsDir)
testFeats, testLabels = loadSet(testSubjects, tasksTest, txtFeatsDir)

print(trainFeats.shape)
print(trainLabels.shape)

print(testFeats.shape)
print(testLabels.shape)

print('Counts')
labelDist = np.bincount(trainLabels)
negProportion = float(labelDist[0]) / float(labelDist[0] + labelDist[1])
posProportion = float(labelDist[1]) / float(labelDist[0] + labelDist[1])
print(np.bincount(trainLabels))
print(np.bincount(testLabels))


model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(K, 4096)))
model.add(Dropout(0.5))

#model.add(MaxPooling1D(pool_length=2))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.25))

model.add(LSTM(1))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy', 'fmeasure'])
print(model.summary())

classWeights = {0: 1-negProportion, 1: 1-posProportion}
print('classWeights', classWeights)
model.fit(trainFeats, trainLabels, nb_epoch=60, batch_size=300, verbose=1, class_weight=classWeights)
#model.fit(trainFeats, trainLabels, nb_epoch=6, batch_size=300, verbose=1, class_weight=classWeights, validation_split=0.5)


print('XXXX Fitting done forward')
# Final evaluation of the model
scores = model.evaluate(testFeats, testLabels, verbose=1)
print(scores)
print(len(scores))


