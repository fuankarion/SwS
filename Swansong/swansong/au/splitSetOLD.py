import numpy


subjects = ['F001', 'F004', 'F007', 'F010', 'F013', 'F016', 'F019', 'F022', 'M002', 
'M005', 'M008', 'M011', 'M014', 'M017', 'F002', 'F005', 'F008', 'F011', 'F014', 
'F017', 'F020', 'F023', 'M003', 'M006', 'M009', 'M012', 'M015', 'M018',
'F003', 'F006', 'F009', 'F012', 'F015', 'F018', 'F021', 'M001', 'M004', 'M007', 
'M010', 'M013', 'M016']

numpy.random.shuffle(subjects)
half = len(subjects) / 2
print(half)
fold1, fold2 = subjects[:half], subjects[half:]

print('fold1', fold1)
print('fold2', fold2)


def handleViews(aTaskdir):
    views = os.listdir(aTaskdir)    
    for aView in views:
        print('add ', aView, ' To set')
        
def handleTasksForSubject(aSubjectDir):
    tasks = os.listdir(aPersonDir)
    for aTask in tasks:
        handleViews(os.path.join(aPersonDir, aTask))
      
def handleSetOrJitter(setPath, subjects):     
    for aSubject in subjects:
        print('process subject', aSubject)
        handleTasksForPerson(os.path.join(setPath, aSubject))

setsOrJittersToAdd=[]


setsOrJittersToAdd.append('/home/afromero/Codes/FERA17/data/Faces/FERA17/BP4D/Train')
for jitterIdx in range(0,30):
    setsOrJittersToAdd.append('/home/afromero/Codes/FERA17/data/Faces/FERA17/BP4D/Jitter/Jitter_'+str(jitterIdx))
    
for aSetOrJitter in setsOrJittersToAdd:
    handleSetOrJitter(aSetOrJitter, fold1)
