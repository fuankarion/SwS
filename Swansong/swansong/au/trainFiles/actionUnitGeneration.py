import random


def gatherTrainSamples(trainFile):
    allPositives = []
    negativeCandidates = []
    with open(trainFile) as f:
        content = f.readlines()
        for aLine in content:
            lineTokens = aLine.split(' ')
                       
            if int(lineTokens[1]) == 1:
                allPositives.append((lineTokens[0], 1))
            else:
                negativeCandidates.append((lineTokens[0], 0))
    
    return allPositives, negativeCandidates

def gatherTrainSamplesForAU(targetAU, trainFile):
    allPositives = []
    negativeCandidates = []
    
    print(trainFile)
    with open(trainFile) as f:
        content = f.readlines()
        for aLine in content:
            lineTokens = aLine.split(' ')

            if int(lineTokens[targetAU + 1]) == 1:
                allPositives.append((lineTokens[0], 1))
            else:
                negativeCandidates.append((lineTokens[0], 0))
    
    return allPositives, negativeCandidates
    
def balanceNegatives(allPositives, negativeCandidates):
    percentage = float(len(allPositives)) / float(len(negativeCandidates))
    
    balancedNegatives = []
    for aNegative in negativeCandidates:
        randomFlag = random.uniform(0, 1)
        if randomFlag <= percentage:
            balancedNegatives.append(aNegative)
    return balancedNegatives
        
def writeSet(tuples, filePath):
    with open(filePath, "a") as myfile:
        for aTuple in tuples:
            #print(' aTuple[0] ', aTuple[0])


            rplc = aTuple[0]
        
            if '/home/afromero/datos/Databases/FERA17/BP4D/Valid' in aTuple[0]:
                #print('Case1')
                rplc = aTuple[0].replace('/home/afromero/datos/Databases/FERA17/BP4D/Valid', '/home/jcleon/Storage/ssd1/FeraData/Valid')
                 
            elif '/home/afromero/datos/Databases/FERA17/BP4D/Train' in aTuple[0]:
                #print('Case2')
                rplc = aTuple[0].replace('/home/afromero/datos/Databases/FERA17/BP4D/Train', '/home/jcleon/Storage/ssd1/FeraData/Train')
               
            myfile.write(rplc + ' ' + str(aTuple[1]) + '\n')

def writeSetFLowVal(tuples, filePath):
    with open(filePath, "a") as myfile:
        for aTuple in tuples:

            rplc = aTuple[0]
        
            if '/home/jcleon/Storage/disk2/FERA17/BP4D/Flow' in aTuple[0]:
                rplc = aTuple[0].replace('/home/jcleon/Storage/disk2/FERA17/BP4D/Flow', '/home/jcleon/Storage/ssd1/FeraData/ValFLow')
              
            myfile.write(rplc + ' ' + str(aTuple[1]) + '\n')
            
def writeSetFLowTrain(tuples, filePath):
    with open(filePath, "a") as myfile:
        for aTuple in tuples:

            rplc = aTuple[0]
        
            if '/home/jcleon/Storage/disk2/FERA17/BP4D/Flow' in aTuple[0]:
                rplc = aTuple[0].replace('/home/jcleon/Storage/disk2/FERA17/BP4D/Flow', '/home/jcleon/Storage/ssd1/FeraData/TrainFlow')
              
            myfile.write(rplc + ' ' + str(aTuple[1]) + '\n')

"""
allPositives, negativeCandidates = gatherTrainSamplesForAU(0)
print('len(allPositives) ', len(allPositives))
print('len(negativeCandidates) ', len(negativeCandidates))

balancedNegatives = balanceNegatives(allPositives, negativeCandidates)
print('len(allPositives) ', len(allPositives))
print('len(balancedNegatives) ', len(balancedNegatives))


finalList=allPositives+balancedNegatives
writeSet(finalList,'/home/jcleon/txt_files/v2/0Label/val.txt')
"""
