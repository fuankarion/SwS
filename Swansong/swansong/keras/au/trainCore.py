import numpy as np
import os
import re
from sklearn.metrics import classification_report
import random


auArray = ['AU01', 'AU10', 'AU12', 'AU04', 'AU15', 'AU17', 'AU23', 'AU14', 'AU06', 'AU07']

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]	

def getCounts(labels):
    labelDist = np.bincount(labels)
    negProportion = float(labelDist[0]) / float(labelDist[0] + labelDist[1])
    posProportion = float(labelDist[1]) / float(labelDist[0] + labelDist[1])
    print(np.bincount(labels))
    return negProportion,posProportion
    
    
def balanceSet(feats, labels, negProportion,thresholdMultiplier):
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
        predsMax.append(np.argmax(feats[anIdx, :]) % 2)
   
    cr = classification_report(gtMax, predsMax)
    print(cr)

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
        
        allfeats[idx, :] = tempFeats
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

    reshaped = np.zeros([len(x)-(K-1), K, feats.shape[1]])
    labelsReshaped = np.zeros(0, dtype=int)
    
    for startIdx in range(0, len(x)-(K-1)):
        targetIndexes = x[startIdx:startIdx + K]  
        midOne = targetIndexes[len(targetIndexes) / 2]
        
        #Feats
        segment = feats[targetIndexes, :]
        reshaped[startIdx, :, :] = segment
        
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
