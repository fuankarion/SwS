import os
import shutil

aus = ['AU01', 'AU04', 'AU06', 'AU07', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU23']
views = ['v1','v2','v3','v4','v5','v6','v7','v8','v9']
rootTrainFiles = '/home/jcleon/Storage/ssd0/fullFaceTrainFiles'


def getUniqueFiles(allFiles):
    seen = set()
    for lst in allFiles:
        if lst not in seen: # filter out previously seen values
            seen.add(lst)                        # add the new values to the set
        if len(seen) % 1000 == 0:
            print('len(seen) ', len(seen)) 

    print('Final len(seen) ', len(seen))    
    return seen

def moveUniqueFiles(uniques, target):
    
    idx=1
    errCount=0
    for aUnique in uniques:
        if 'Jitter' in aUnique:
            idx=idx+1
            aUnique = aUnique.replace('/home/afromero/Codes/FERA17/data/Faces/FERA17/BP4D/Jitter', '/home/jcleon/Storage/disk2/Jitter')
            target = aUnique.replace('/home/jcleon/Storage/disk2/Jitter', '/home/jcleon/Storage/disk0/Jitter')
            
            if os.path.exists(target):
                continue
            #print('Move ', aUnique, ' To ', target)
            
            superDir = target[:-8]
            if not os.path.exists(superDir):
		os.makedirs(superDir)
                
            try:
                shutil.copy(aUnique, target)
            except:
                #print('Err with ', aUnique)
                errCount=errCount+1
            
            if idx%500==0:
                print('idx',idx)
                print('errCount',errCount)
            

for aView in views:
    allFiles = []
    for anAU in aus:
        auOhne0 = anAU.replace('0', '')
        checkFile = rootTrainFiles + '/' + aView + '/Training_' + auOhne0 + '.txt'
        print('checkFile', checkFile)
        
        with open(checkFile) as f:
            content = f.readlines()
            for aLine in content:
                lineTokens = aLine.split(' ')
                allFiles.append(lineTokens[0])
    print('allFiles ', len(allFiles))
    uniqFiles = getUniqueFiles(allFiles)
    moveUniqueFiles(uniqFiles, 'None')
    

    