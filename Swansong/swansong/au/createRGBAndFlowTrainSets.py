import os
import random
import sys

sys.path.append('../')

from actionUnitGeneration import gatherTrainSamplesForAU
from actionUnitGeneration import balanceNegatives

def writeSetToFile(aSet, targetFile):
    with open(targetFile, 'a') as myfile:
        idx = 0
        for aLine in aSet:
            
            if idx % 1000 == 0:
                print('Writing ', idx, ' lines') 
            myfile.write(aLine[0] + ' ' + str(aLine[1]) + '\n')    
            idx = idx + 1

def writeLinesToFile(lines, targetFile):
    with open(targetFile, 'a') as myfile:
        idx = 0
        for aLine in lines:
            if idx % 1000 == 0:
                print('Writing ', str(idx), ' lines') 
            myfile.write(aLine )    
            idx = idx + 1


def shuffleFilesAlike(file1, file2):
    randomIdx = shuffleFileIndexes(file1)
    return shuffleFilesByRandomIdx(file1, file2, randomIdx)
    
    
def shuffleFileIndexes(sourceFile1):   
    randomIndexes = []

    currentLine = 0
    with open(sourceFile1, 'r') as myfile:
        content = myfile.readlines()
        for aLine in content:
            randomIndexes.append(currentLine)
            currentLine = currentLine + 1
            
    random.shuffle(randomIndexes)
    return randomIndexes

def shuffleFilesByRandomIdx(sourceFile1, sourceFile2, randomIndexes): 
    shuffledFile1 = [None] * len(randomIndexes)
    shuffledFile2 = [None] * len(randomIndexes)
    
    lineIdx = 0
    with open(sourceFile1, 'r') as myfile:
        content = myfile.readlines()
        for aLine in content:
            shuffledFile1[randomIndexes[lineIdx]] = aLine
            lineIdx = lineIdx + 1

    lineIdx = 0
    with open(sourceFile2, 'r') as myfile:
        content = myfile.readlines()
        for aLine in content:
            shuffledFile2[randomIndexes[lineIdx]] = aLine
            lineIdx = lineIdx + 1
            
    return shuffledFile1, shuffledFile2    
	
def generateFlowSetFromRGBSet(fileRGB, sameDirectoryDepth, flowBasePath):
    flowSet = []
    with open(fileRGB, 'r') as myfile:
        content = myfile.readlines()
        for aLine in content:

            tokensPaths = aLine.split('/')
            pathSplit = '/'.join(tokensPaths[:sameDirectoryDepth]), '/'.join(tokensPaths[sameDirectoryDepth:])    
           
            flowFullPath = os.path.join(flowBasePath, pathSplit[1])
            flowFullPathTokens = flowFullPath.split(' ')
            flowSet.append((flowFullPathTokens[0], flowFullPathTokens[1][:-1]))
    return flowSet

def createFullSetForLabel(auTarget, viewsDirectory, actualView):
    trainFile = os.path.join(viewsDirectory, actualView + '/Training.txt')
    valFile = os.path.join(viewsDirectory, actualView + '/Test.txt')
    
    allPositivesTrain, negativeCandidatesTrain = gatherTrainSamplesForAU(auTarget, trainFile)
    balancedNegativesTrain = balanceNegatives(allPositivesTrain, negativeCandidatesTrain)
    
    allPositivesVal, negativeCandidatesVal = gatherTrainSamplesForAU(auTarget, valFile)
    balancedNegativesVal = balanceNegatives(allPositivesVal, negativeCandidatesVal)
        
    finalTrainSet = allPositivesTrain + balancedNegativesTrain
    finalValSet = allPositivesVal + balancedNegativesVal
    return finalTrainSet, finalValSet

#EXEC
finalTrainSet, finalValSet = createFullSetForLabel(3, '/home/jcleon/Storage/ssd0/fullFaceTrainFiles', 'v3')
writeSetToFile(finalTrainSet, '/home/jcleon/Storage/ssd1/au3V3BalancedSets/train.txt')
writeSetToFile(finalValSet, '/home/jcleon/Storage/ssd1/au3V3BalancedSets/val.txt')

flowSetTrain = generateFlowSetFromRGBSet('/home/jcleon/Storage/ssd1/au3V3BalancedSets/train.txt', 7, '/home/jcleon/Storage/ssd1/Flow/Train')
flowSetVal = generateFlowSetFromRGBSet('/home/jcleon/Storage/ssd1/au3V3BalancedSets/val.txt', 7, '/home/jcleon/Storage/ssd1/Flow/Val')

writeSetToFile(flowSetTrain, '/home/jcleon/Storage/ssd1/au3V3BalancedSets/trainFlow.txt')
writeSetToFile(flowSetVal, '/home/jcleon/Storage/ssd1/au3V3BalancedSets/valFlow.txt')

shuffledTrain, shuffledTrainFlow = shuffleFilesAlike('/home/jcleon/Storage/ssd1/au3V3BalancedSets/train.txt', '/home/jcleon/Storage/ssd1/au3V3BalancedSets/trainFlow.txt')
shuffledVal, shuffledValFlow = shuffleFilesAlike('/home/jcleon/Storage/ssd1/au3V3BalancedSets/val.txt', '/home/jcleon/Storage/ssd1/au3V3BalancedSets/valFlow.txt')

writeLinesToFile(shuffledTrainFlow, '/home/jcleon/Storage/ssd1/au3V3BalancedSets/shuffled/trainFlow.txt')
writeLinesToFile(shuffledValFlow, '/home/jcleon/Storage/ssd1/au3V3BalancedSets/shuffled/valFlow.txt')
writeLinesToFile(shuffledTrain, '/home/jcleon/Storage/ssd1/au3V3BalancedSets/shuffled/train.txt')
writeLinesToFile(shuffledVal, '/home/jcleon/Storage/ssd1/au3V3BalancedSets/shuffled/val.txt')