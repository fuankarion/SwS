##generate text train files from directory hierachy
import os
import re 
from string import Template
from subprocess import call

lmdbCommandTemplate = Template('GLOG_logtostderr=1 /home/jcleon/Software/caffe/build/tools/convert_imageset --resize_height=256 --resize_width=256 --encoded /  $trainFile  $lmdbFile ')

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]	


def createTrainFileFromVid(targetPath, videoPath):
    allFilesInDir = os.listdir(videoPath)
    allFilesInDirSorted = sorted(allFilesInDir, key=natural_key)
    lineData = []
    vidPathTokens = videoPath.split('/')
    label = vidPathTokens[-1]
    label = label[-1]
    
    for aFile in allFilesInDirSorted:
        fullPath = os.path.join(videoPath, aFile)
        lineData.append(fullPath + ' ' + label)

    with open(targetPath, "a") as myfile:
        for aLine in lineData:
            myfile.write(aLine + '\n')
    
    
def createLDMBFromDir(videoPath, txtFilesPath, lmdbsPath):
    dirPathTokens = videoPath.split('/')
    subject = dirPathTokens[-3]
    task = dirPathTokens[-2]
    vid = dirPathTokens[-1]
    
    targetTrainFile = os.path.join(txtFilesPath, subject + '-' + task + '-' + vid)
    targetLMB = os.path.join(lmdbsPath, subject + '-' + task + '-' + vid)
    
    print('targetTrainFile ', targetTrainFile)
    print('targetLMB ', targetLMB)
    
    createTrainFileFromVid(targetTrainFile,videoPath)
    
    command = lmdbCommandTemplate.substitute(trainFile=targetTrainFile, lmdbFile=targetLMB) 
    print('lmdbcommand ',command)
    call(command, shell=True)

def handleViews(aTaskdir):
    print('aTaskdir ', aTaskdir)
    
    views = os.listdir(aTaskdir)    
    for aView in views:
        print(' ')
        print('create LMDB for ', os.path.join(aTaskdir, aView))
        createLDMBFromDir(os.path.join(aTaskdir, aView), '/home/jcleon/Storage/disk2/FERA17/BP4D/forward/txt', '/home/jcleon/Storage/disk2/FERA17/BP4D/forward/lmdb')
        
def handleTasksForPerson(aPersonDir):
    tasks = os.listdir(aPersonDir)

    for aTask in tasks:
        handleViews(os.path.join(aPersonDir, aTask))
      
def handleSubjects(rootPath,): 
    persons = os.listdir(rootPath)
    print('len(dirs)', len(persons))
    for aPerson in persons:
        print('process person', aPerson)
        handleTasksForPerson(os.path.join(rootPath, aPerson))

#root of the directory hierachy
root = '/home/jcleon/Storage/disk2/FERA17/BP4D/Valid'

handleSubjects(root)
    
