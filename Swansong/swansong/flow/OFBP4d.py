import os
from shutil import copyfile
from string import Template
import subprocess

binaryPath = './broxDir'
processes = set()
max_processes = 50

def issueFlowCommandToPool(inputDir, outputDir):   
    actualCMD = Template('$bnPath --source $sDir --ziel $zDir ').substitute(bnPath=binaryPath, sDir=inputDir, zDir=outputDir) 
    print('Issuing to pool', actualCMD)
    processes.add(subprocess.Popen(actualCMD, shell=True))
    if len(processes) >= max_processes:
        os.wait()
        processes.difference_update([p for p in processes if p.poll() is not None])
        
    return 0#No idea we need this?

def fillFinalImageInDir(path):
    print('Fix dir ', path)
    num_files = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    try:
        copyfile(path + '/' + str(num_files-1).zfill(4) + '.jpg', path + '/' + str(num_files + 1).zfill(4) + '.jpg')
    except:
        try:
            copyfile(path + '/' + str(num_files-1).zfill(3) + '.jpg', path + '/' + str(num_files + 1).zfill(3) + '.jpg')
        except:
            try:
                copyfile(path + '/' + str(num_files-1).zfill(0) + '.jpg', path + '/' + str(num_files + 1).zfill(0) + '.jpg')
            except:
                print('Ignore')
def calculateOFBroxGPUAnSet(sourceDir, targetDir, original):    
    #print('calculateOFBroxGPUAnSet')   
    #print('sourceDir ',sourceDir)
    #print('targetDir ',targetDir)
        
    if not os.path.exists(targetDir):
        os.makedirs(targetDir)
        print ('created ', targetDir)

    #if os.path.isfile(targetDir + '/0001.jpg'):#frames are there already
        #print ('Frames for  ', targetDir, ' OF already calculated')
        #fillFinalImageInDir(targetDir)
        #return
    numFilesSource = len([f for f in os.listdir(sourceDir) if os.path.isfile(os.path.join(sourceDir, f))])
    numFilesTarget = len([f for f in os.listdir(targetDir) if os.path.isfile(os.path.join(targetDir, f))])
    numfilesOriginal = len([f for f in os.listdir(original) if os.path.isfile(os.path.join(original, f))])	 

    if numfilesOriginal != numFilesSource or numfilesOriginal != numFilesTarget:
        print('numFilesSource', numFilesSource)
        print('numFilesTarget', numFilesTarget)
        print('numfilesOriginal', numfilesOriginal)
        print('original ', original)

    if numFilesSource > numFilesTarget + 1:
        issueFlowCommandToPool(sourceDir, targetDir)

    elif numFilesSource == numFilesTarget + 1:
	fillFinalImageInDir(targetDir)

        
def handleTasksForPerson(aPersonDir, targetPerson, original):
    tasks = os.listdir(aPersonDir)

    for aTask in tasks:
        #handleViews(os.path.join(aPersonDir, aTask), os.path.join(targetPerson, aTask))
 	calculateOFBroxGPUAnSet(os.path.join(aPersonDir, aTask), os.path.join(targetPerson, aTask), os.path.join(original, aTask))
      
def handleSubjects(rootPath, targetRoot, original): 
    persons = os.listdir(rootPath)
    #print('len(dirs)', len(persons))
    for aPerson in persons:
        #print('process person', aPerson)
        handleTasksForPerson(os.path.join(rootPath, aPerson), os.path.join(targetRoot, aPerson), os.path.join(original, aPerson))
          
#root of the directory hierachy

sourcePath = '/home/afromero/datos/Databases/BP4D/SequencesResized'
targetBase = '/home/afromero/datos/Databases/BP4D/SequencesResizedFlow'
original = '/home/afromero/datos/Databases/BP4D/Sequences'


handleSubjects(sourcePath, targetBase, original)

####handleSubjects(original, targetBase,sourcePath)

    
