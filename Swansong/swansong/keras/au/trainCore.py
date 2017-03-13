import numpy as np
import os
import random
import re
from sklearn.metrics import classification_report
from keras.callbacks import *
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Merge
from keras.models import Sequential
from keras.optimizers import SGD
import numpy as np
from sklearn.metrics import classification_report
from trainCore import *


auArray = ['AU01', 'AU10', 'AU12', 'AU04', 'AU15', 'AU17', 'AU23', 'AU14', 'AU06', 'AU07']

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]	

def getCounts(labels):
    labelDist = np.bincount(labels)
    negProportion = float(labelDist[0]) / float(labelDist[0] + labelDist[1])
    posProportion = float(labelDist[1]) / float(labelDist[0] + labelDist[1])
    print(np.bincount(labels))
    return negProportion, posProportion
    
    
def balanceSet(feats, labels, negProportion, thresholdMultiplier):
    deleteIdx = []
    for rowDeleteIdx in range(0, labels.shape[0]):
        if labels[rowDeleteIdx] == 0:
            randomFlag = random.uniform(0, 1)
            if randomFlag <= negProportion * thresholdMultiplier:
                deleteIdx.append(rowDeleteIdx)

    balancedFeats = np.delete(feats, deleteIdx, 0)
    balancedLabels = np.delete(labels, deleteIdx, 0)
    
    return balancedFeats, balancedLabels


def getClassificationScoreMaxCriteria(feats, labels):
    gtMax = []
    predsMax = []
    for anIdx in range(0, feats.shape[0]):
        gtMax.append(labels[anIdx])
        predsMax.append(np.argmax(feats[anIdx,:]) % 2)
   
    cr = classification_report(gtMax, predsMax)
    print(cr)
    return predsMax,cr

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
   
def loadFeats(rootPath, aSubject, aTask, view, K, bulkLoadDir, extra, au):
    bulkLoadFile = bulkLoadDir + '/' + aSubject + '-' + aTask + '-' + view + '-' + extra  + str(K) + '.npy'
    data, flag = testBulkLoadData(bulkLoadFile)
    if flag:
        return data, flag
    
    txtFilesPath = rootPath + '/' + aSubject + '/' + aTask + '/' + view + '/'
    #print('txtFilesPath ', txtFilesPath)
    
    allFilesInDir = os.listdir(txtFilesPath)
    allFilesInDir = sorted(allFilesInDir, key=natural_key)
    allFilesInDir = allFilesInDir[:-1]#pesky label file
    
    allfeats = np.zeros([len(allFilesInDir), 2])#TODO hardcoded 2
    
    idx = 0
    for aFile in allFilesInDir:
        tempPath = os.path.join(txtFilesPath, aFile)
        tempFeats = np.loadtxt(tempPath)
        
        allfeats[idx,:] = tempFeats
        idx = idx + 1
        if idx % 100 == 0:
            print(idx, '/', len(allFilesInDir))
        
    return allfeats, False

def loadLabels(rootPath, aSubject, aTask, view, K, bulkLoadDir, extra, au):
    bulkLoadFile = bulkLoadDir + '/' + aSubject + '-' + aTask + '-' + view + '-' + extra  + str(K) + '.npy'
    data, flag = testBulkLoadData(bulkLoadFile)
    if flag:
        return data, flag
    
    
    txtFilesPath = rootPath + '/' + aSubject + '/' + aTask + '/' + view + '/'
    #print('txtFilesPath ', txtFilesPath)
    
    labelFile = os.path.join(txtFilesPath, 'labels.txt')
    auIdx = auArray.index(au)
    labels = []
    with open(labelFile) as f:
        content = f.readlines()
        for aLine in content:
            lineTokens = aLine.split(',')
            label = lineTokens[auIdx + 1]
            
            labels.append(int(label))
    return np.array(labels), False
            
            
#map to ?,K,2
def reshapeTrainData(feats, labels, K):
    print('Reshape')
    x = range(0, feats.shape[0])
    
    #print('x',x)
    #print('K',K)

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

def loadSet(subjects, tasks, txtDir, view, K, bulkLoadDir, au):
    trainFeats = None
    trainLabels = None
    
    debugFeats = None
    debuglabels = None
    for aSubject in subjects:
        for aTask in tasks:
            print(' ')
            print('Subject ', aSubject)
            print('Task ', aTask)

            try:                   
                labels, bulkLoadLabels = loadLabels(txtDir, aSubject, aTask, view, K, bulkLoadDir, 'LabelsK', au)    
                feats, bulkLoadFeats = loadFeats(txtDir, aSubject, aTask, view, K, bulkLoadDir, 'FeatsK', au)

            except Exception  as err:
                print(err)
                print('Skip load ', aSubject, ' ', aTask)
                continue
                        
            if not bulkLoadFeats:
                featsReshaped, labelsReshaped = reshapeTrainData(feats, labels, K)

                #Save to bulk load
                writeBulkLoadFeats(bulkLoadDir, aSubject, aTask, view, K, featsReshaped, 'FeatsK')
                writeBulkLoadFeats(bulkLoadDir, aSubject, aTask, view, K, labelsReshaped, 'LabelsK')
                
                if debugFeats == None:
                    debugFeats = np.copy(feats)
                else:
                    debugFeats = np.concatenate((debugFeats, feats), axis=0)

                if debuglabels == None:
                    debuglabels = np.copy(labels)
                else:
                    debuglabels = np.concatenate((debuglabels, labels), axis=0)
            
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
    
    return trainFeats, trainLabels


def rebalanceFeats():
    #get and balance train set
    
    print('Unablanced samples')
    negProportion, posProportion = getCounts(trainLabelsUnbalanced)
    trainFeats, trainLabels = balanceSet(trainFeatsUnbalanced, trainLabelsUnbalanced, negProportion, 0.35)
    #Recalc
    print('Rebalanced samples')
    negProportion, posProportion = getCounts(trainLabels)
    
def trainLSTMModel(au, view, trainFold, jointModel, steps,baseFeatsDir):

    txtFeatsTrainDir = baseFeatsDir + '/' + au + '_' + view + '_Fold' + str(trainFold)

    bulkLoadDirTrain = '/home/jcleon/Storage/ssd0/BL/bulkLoad' + au + view + 'fold' + str(trainFold) + '/Train'
    if not os.path.exists(bulkLoadDirTrain):
        os.makedirs(bulkLoadDirTrain)
        
   

    trainSubject = None
    if trainFold == 0:
        trainSubjects = ['M012', 'M017', 'M006', 'M008', 'M002', 'F016', 'F011', 'M015', 'M016', 'F006', 'M011', 'F008', 'F003', 'M009', 'F002', 'M003', 'M014', 'F023', 'F015', 'F021']
    if trainFold == 1:
        trainSubjects = ['F022', 'F013', 'F010', 'F007', 'M005', 'F012', 'F019', 'F005', 'M004', 'M018', 'F020', 'F001', 'F014', 'F004', 'F017', 'F009', 'M001', 'M010', 'M007', 'M013', 'F018']

    tasksTrain = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'T13', 'T14', 'T15']
    

    #get and balance train set
    trainFeatsUnbalanced, trainLabelsUnbalanced = loadSet(trainSubjects, tasksTrain, txtFeatsTrainDir, view, steps, bulkLoadDirTrain, au)
    print('Unablanced samples')
    negProportion, posProportion = getCounts(trainLabelsUnbalanced)

    jointModel.fit([trainFeatsUnbalanced, trainFeatsUnbalanced], trainLabelsUnbalanced, nb_epoch=150, batch_size=12000, verbose=1)
    return jointModel, trainLabelsUnbalanced, trainFeatsUnbalanced

def evalModel(au, view, evalFold, trainedModel, trainLabelsUnbalanced, trainFeatsUnbalanced, steps,baseFeatsDir):
    gtMax = []
    predsMax = []
    for anIdx in range(0, trainLabelsUnbalanced.shape[0]):
        gtMax.append(trainLabelsUnbalanced[anIdx])
        predsMax.append(np.argmax(trainFeatsUnbalanced[anIdx, 0, :]) % 2)
    
    #Val subjects
    testSubjects = ['F007', 'F008', 'F009', 'F010', 'F011', 'M001', 'M002', 'M003', 'M004', 'M005', 'M006', 'rF001', 'rF002', 'rM001', 'rM002', 'rM003', 'rM004', 'rM005', 'rM006', 'rM007']
    tasksVal = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'T13', 'T14', 'T15']
    
    bulkLoadDirVal = '/home/jcleon/Storage/ssd0/BL/bulkLoad' + au + view + 'fold' + str(evalFold) + '/Val'
    if not os.path.exists(bulkLoadDirVal):
        os.makedirs(bulkLoadDirVal)
    txtFeatsTestDir = baseFeatsDir + '/' + au + '_' + view + '_Fold' + str(evalFold)
    testFeats, testLabels = loadSet(testSubjects, tasksVal, txtFeatsTestDir, view, steps, bulkLoadDirVal, au)

    result=''
    print('Counts Train Full')
    negProportion, posProportion = getCounts(trainLabelsUnbalanced)
    #print('Classification MAX Score Train Full')
    dontCare,cr=getClassificationScoreMaxCriteria(trainFeatsUnbalanced, trainLabelsUnbalanced)
    result=result+'Classification MAX Score Train Full'+'\n'
    result=result+cr+'\n'

    print('LSTM Classficiation Train Set')
    preds = trainedModel.predict_classes([trainFeatsUnbalanced, trainFeatsUnbalanced])
    #print(' ')
    #print(classification_report(trainLabelsUnbalanced, preds))
    result=result+'LSTM Classficiation Train Se'+'\n'
    result=result+classification_report(trainLabelsUnbalanced, preds)+'\n'

    print('LSTM Classficiation Val Set')
    preds = trainedModel.predict_classes([testFeats, testFeats])
    #print(' ')
    #print(classification_report(testLabels, preds))
    result=result+'LSTM Classficiation Val Set'+'\n'
    result=result+classification_report(testLabels, preds)+'\n'

    print('Counts Val UNBalanced')
    getCounts(testLabels)
    #print('Classification MAX Score Val UNBalanced')
    getClassificationScoreMaxCriteria(testFeats, testLabels)
    
    dontCare,cr=getClassificationScoreMaxCriteria(testFeats, testLabels)
    result=result+'Classification MAX Score Val UNBalanced'+'\n'
    result=result+cr+'\n'
    
    return result

def createModel(timeSteps):
    ###Kerasmodel
    reluBranch = Sequential()
    reluBranch.add(Dense(2, activation='relu', input_shape=(timeSteps, 2), init='uniform'))
    reluBranch.add(Dense(16, activation='sigmoid', init='uniform'))
    reluBranch.add(Dense(8, activation='relu', init='uniform'))

    sigmoidBranch = Sequential()
    sigmoidBranch.add(Dense(2, activation='sigmoid', input_shape=(timeSteps, 2), init='uniform'))
    sigmoidBranch.add(Dense(16, activation='relu', init='uniform'))
    sigmoidBranch.add(Dense(8, activation='sigmoid', init='uniform'))

    merged = Merge([reluBranch, sigmoidBranch], mode='concat')

    jointModel = Sequential()
    jointModel.add(merged)
    jointModel.add(LSTM(10))
    jointModel.add(Dense(1, activation='sigmoid'))

    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    jointModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'fmeasure'])
    print(jointModel.summary())
    return jointModel
