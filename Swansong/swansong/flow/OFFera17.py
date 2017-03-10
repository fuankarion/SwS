import os
from shutil import copyfile
from string import Template
import subprocess

binaryPath = '/home/jcleon/DAKode/BroxOFGPU2/broxDir'
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
        print('Ignore')
        
        
def calculateOFBroxGPUAnSet(sourceDir, targetDir):    
    print('calculateOFBroxGPUAnSet')   
    print('sourceDir ', sourceDir)
    print('targetDir ', targetDir)
        
    if not os.path.exists(targetDir):
        os.makedirs(targetDir)
        print ('created ', targetDir)

    numFilesSource = len([f for f in os.listdir(sourceDir) if os.path.isfile(os.path.join(sourceDir, f))])
    numFilesTarget = len([f for f in os.listdir(targetDir) if os.path.isfile(os.path.join(targetDir, f))])

    if numFilesSource != numFilesTarget:
        print('numFilesSource', numFilesSource)
        print('numFilesTarget', numFilesTarget)

    if numFilesSource > numFilesTarget + 1:
        issueFlowCommandToPool(sourceDir, targetDir)
    elif numFilesSource == numFilesTarget + 1:#Just Fix
	fillFinalImageInDir(targetDir)
    else:
        print('Done with ',sourceDir)


def handleViews(taskSourceDir, taskTargetDir):
    views = os.listdir(taskSourceDir)
    
    for aView in views:
        print('Process view', aView)
        viewSourceDir = os.path.join(taskSourceDir, aView)
        viewTargetDir = os.path.join(taskTargetDir, aView)

        calculateOFBroxGPUAnSet(viewSourceDir, viewTargetDir)
        
                
def handleTasksForPerson(personSoruceDir, personTargetDir):
    tasks = os.listdir(personSoruceDir)

    for aTask in tasks:
        print('Process task', aTask)
        handleViews(os.path.join(personSoruceDir, aTask), os.path.join(personTargetDir, aTask))
 	
      
def handleSubjects(sourceRoot, targetRoot): 
    persons = os.listdir(sourceRoot)
    for aPerson in persons:
        print('Process person', aPerson)
        handleTasksForPerson(os.path.join(sourceRoot, aPerson), os.path.join(targetRoot, aPerson))
          
#root of the directory hierachy

"""
sourcePath = '/home/jcleon/Storage/disk2/resizedFera17-512/Valid'
targetBase = '/home/jcleon/Storage/disk2/resizedFera17-512Flow/Valid'
handleSubjects(sourcePath, targetBase)
"""


sourcePath = '/home/jcleon/Storage/disk2/Jitter/Jitter_0'
targetBase = '/home/jcleon/Storage/disk2/Jitter/Jitter_0Flow'
handleSubjects(sourcePath, targetBase)


sourcePath = '/home/jcleon/Storage/disk2/Jitter/Jitter_1'
targetBase = '/home/jcleon/Storage/disk2/JitterFlow/Jitter_1'
handleSubjects(sourcePath, targetBase)

sourcePath = '/home/jcleon/Storage/disk2/Jitter/Jitter_2'
targetBase = '/home/jcleon/Storage/disk2/JitterFlow/Jitter_2'
handleSubjects(sourcePath, targetBase)

sourcePath = '/home/jcleon/Storage/disk2/Jitter/Jitter_4'
targetBase = '/home/jcleon/Storage/disk2/JitterFlow/Jitter_4'
handleSubjects(sourcePath, targetBase)

sourcePath = '/home/jcleon/Storage/disk2/Jitter/Jitter_5'
targetBase = '/home/jcleon/Storage/disk2/JitterFlow/Jitter_5'
handleSubjects(sourcePath, targetBase)

sourcePath = '/home/jcleon/Storage/disk2/Jitter/Jitter_6'
targetBase = '/home/jcleon/Storage/disk2/JitterFlow/Jitter_6'
handleSubjects(sourcePath, targetBase)
    
