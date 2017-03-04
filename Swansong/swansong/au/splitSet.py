import numpy
import os

subjects = ['F001', 'F004', 'F007', 'F010', 'F013', 'F016', 'F019', 'F022', 'M002', 
'M005', 'M008', 'M011', 'M014', 'M017', 'F002', 'F005', 'F008', 'F011', 'F014', 
'F017', 'F020', 'F023', 'M003', 'M006', 'M009', 'M012', 'M015', 'M018',
'F003', 'F006', 'F009', 'F012', 'F015', 'F018', 'F021', 'M001', 'M004', 'M007', 
'M010', 'M013', 'M016']

numpy.random.shuffle(subjects)
half = len(subjects) / 2
print(half)
fold0, fold1 = subjects[:half], subjects[half:]

print('fold0', fold0)
print('fold1', fold1)

#this is only for correction
"""
fold0 = ['M012', 'M017', 'M006', 'M008', 'M002', 'F016', 'F011', 'M015', 'M016', 'F006', 'M011', 'F008', 'F003', 'M009', 'F002', 'M003', 'M014', 'F023', 'F015', 'F021']
fold1 = ['F022', 'F013', 'F010', 'F007', 'M005', 'F012', 'F019', 'F005', 'M004', 'M018', 'F020', 'F001', 'F014', 'F004', 'F017', 'F009', 'M001', 'M010', 'M007', 'M013', 'F018']
"""
def lineBelongsToFold(line, fold0, fold1):
    for aSubjectName in fold0:
        if aSubjectName in line:
            return 0
    for aSubjectName in fold1:
        if aSubjectName in line:
            return 1
    
    print('XXXX this is an error')
    return 2
    
def getStats(fold):
    numPositives = 0
    numNegatives = 0
    for aLine in fold:
        lineTokens = aLine.split(' ')
        if lineTokens[1][0] == '1':
            numPositives = numPositives + 1
        if lineTokens[1][0] == '0':
            numNegatives = numNegatives + 1
        
    return float(numNegatives) / float(numPositives)
    
def splitFile(filePath):
    
    fold0Set = []
    fold1Set = []
    fileQuant = 0
    with open(filePath) as f:
        content = f.readlines()
        for aLine in content:
            fileQuant = fileQuant + 1
            foldBelongs = lineBelongsToFold(aLine, fold0, fold1)
            if foldBelongs == 0:
                fold0Set.append(aLine)
                continue
            if foldBelongs == 1:
                fold1Set.append(aLine)
                continue
            print('XXXX this is an error')
                
    print('fileQuant ', fileQuant)
    print('fold0Set ', len(fold0Set))
    print('fold1Set ', len(fold1Set))
            
    
    return fold0Set, fold1Set, getStats(fold0Set), getStats(fold1Set), fileQuant, len(fold0Set), len(fold1Set)
            
def writeLinesToFile(lineSet, targetFile):
    with open(targetFile, 'a') as myfile:
        idx = 0
        for aLine in lineSet:
            if idx % 10000 == 0:
                print('Writing ', idx, ' lines') 
            myfile.write(aLine)    
            idx = idx + 1
            
au = ['AU1', 'AU10', 'AU12', 'AU4', 'AU15', 'AU17', 'AU23', 'AU14', 'AU6', 'AU7']   
#Only for correction
#au = ['AU4', 'AU14',]   
tagetDir = '/home/jcleon/testSplitsCorrection/'
for anAU in au:
    for aView in range(1, 10):
        fileTarget = '/home/afromero/Codes/FERA17/data/txt_files/RGB/v' + str(aView) + '/Training_' + anAU + '.txt'
        print(fileTarget)
        fold0Set, fold1Set, stats0, stats1, lenFull, lenFold0, lenFold1 = splitFile(fileTarget)
        
        pathFold0 = tagetDir + '/fold0/v' + str(aView) 
        pathFold1 = tagetDir + '/fold1/v' + str(aView) 
        
        statsLines = []
        statsLines.append('v' + str(aView) + ',' + anAU + ',' + str(stats0) + ',' + str(stats1) + ',' + str(lenFull) + ',' + str(lenFold0) + ',' + str(lenFold1) + '\n')
        
        if not os.path.exists(pathFold0):
            os.makedirs(pathFold0)
            
        if not os.path.exists(pathFold1):
            os.makedirs(pathFold1)
        
        writeLinesToFile(fold0Set, pathFold0 + '/Training_' + anAU + '.txt')
        writeLinesToFile(fold1Set, pathFold1 + '/Training_' + anAU + '.txt')
        
        writeLinesToFile(statsLines, tagetDir + '/stats.txt')
        