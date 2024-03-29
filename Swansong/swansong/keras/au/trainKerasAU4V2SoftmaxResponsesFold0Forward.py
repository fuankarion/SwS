from keras.callbacks import *
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Merge
from keras.models import Sequential
from keras.optimizers import SGD
import numpy as np
from sklearn.metrics import classification_report
from trainCore import *

K = 3
au = 'AU04'
view = 'v2'
thersholdBalance = 0.05

"""
txtFeatsTrainDir = '/home/jcleon/Storage/ssd0/FeatsTrain/SoftMaxActivations/AU01_v1_Fold0'
txtFeatsTestDir = '/home/jcleon/Storage/ssd0/FeatsVal/SoftMaxActivations/AU01_v1_Fold0'
"""

txtFeatsTrainDir = '/home/jcleon/Storage/ssd0/FeatsTrain/SoftMaxActivations/' + au + '_' + view + '_Fold1'
txtFeatsTestDir = '/home/jcleon/Storage/ssd0/FeatsVal/SoftMaxActivations/' + au + '_' + view + '_Fold1'

bulkLoadDirTrain = '/home/jcleon/Storage/ssd0/bulkLoadAU4V2TrainBINARY/Train'
bulkLoadDirVal = '/home/jcleon/Storage/ssd0/bulkLoadAU4V2ValBINARY/Val'

trainSubjectsFold0 = ['M012', 'M017', 'M006', 'M008', 'M002', 'F016', 'F011', 'M015', 'M016', 'F006', 'M011', 'F008', 'F003', 'M009', 'F002', 'M003', 'M014', 'F023', 'F015', 'F021']
#trainSubjectsFold1 = ['F022', 'F013', 'F010', 'F007', 'M005', 'F012', 'F019', 'F005', 'M004', 'M018', 'F020', 'F001', 'F014', 'F004', 'F017', 'F009', 'M001', 'M010', 'M007', 'M013', 'F018']
#fullTrainSubjects = ['M012', 'M017', 'M006', 'M008', 'M002', 'F016', 'F011', 'M015', 'M016', 'F006', 'M011', 'F008', 'F003', 'M009', 'F002', 'M003', 'M014', 'F023', 'F015', 'F021', 'F022', 'F013', 'F010', 'F007', 'M005', 'F012', 'F019', 'F005', 'M004', 'M018', 'F020', 'F001', 'F014', 'F004', 'F017', 'F009', 'M001', 'M010', 'M007', 'M013', 'F018']

#Val subjects
testSubjects = ['F007', 'F008', 'F009', 'F010', 'F011', 'M001', 'M002', 'M003', 'M004', 'M005', 'M006', 'rF001', 'rF002', 'rM001', 'rM002', 'rM003', 'rM004', 'rM005', 'rM006', 'rM007']

tasksTrain = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'T13', 'T14', 'T15']
tasksTest = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'T13', 'T14', 'T15']

#get and balance train set
trainFeatsUnbalanced, trainLabelsUnbalanced = loadSet(trainSubjectsFold0, tasksTrain, txtFeatsTrainDir, view, K, bulkLoadDirTrain, au)
print('Unablanced samples')
negProportion, posProportion = getCounts(trainLabelsUnbalanced)
trainFeats, trainLabels = balanceSet(trainFeatsUnbalanced, trainLabelsUnbalanced, negProportion, thersholdBalance)
#Recalc
print('Rebalanced samples')
negProportion, posProportion = getCounts(trainLabels)

###Kerasmodel
reluBranch = Sequential()
reluBranch.add(Dense(2, activation='relu', input_shape=(K, 2), init='uniform'))
reluBranch.add(Dense(16, activation='sigmoid', init='uniform'))
reluBranch.add(Dense(8, activation='relu', init='uniform'))

sigmoidBranch = Sequential()
sigmoidBranch.add(Dense(2, activation='sigmoid', input_shape=(K, 2), init='uniform'))
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

classWeights = {0: (1-negProportion) * 1, 1: (1-posProportion) * 1}
print('classWeights', classWeights)
#model.fit(trainFeats, trainLabels, nb_epoch=150, batch_size=2000, verbose=1, class_weight=classWeights)
jointModel.fit([trainFeats, trainFeats], trainLabels, nb_epoch=150, batch_size=6000, verbose=1)

gtMax = []
predsMax = []
for anIdx in range(0, trainLabels.shape[0]):
    #print(trainFeats[anIdx,0,:],' max ', np.argmax(trainFeats[anIdx,0,:]),' gt ',testLabels[anIdx])
    gtMax.append(trainLabels[anIdx])
    predsMax.append(np.argmax(trainFeats[anIdx, 0, :]) % 2)
    #print(trainLabels[anIdx,1,:])
    
testFeats, testLabels = loadSet(testSubjects, tasksTest, txtFeatsTestDir, view, K, bulkLoadDirVal, au)

print('Counts Train Full')
negProportion, posProportion = getCounts(trainLabelsUnbalanced)
print('Classification MAX Score Train Full')
getClassificationScoreMaxCriteria(trainFeatsUnbalanced, trainLabelsUnbalanced)

print('Counts Train Balanced')
getCounts(trainLabels)
print('Classification MAX Score Train Balanced')
getClassificationScoreMaxCriteria(trainFeats, trainLabels)

print('LSTM Classficiation Train Set')
preds = jointModel.predict_classes([trainFeats, trainFeats])
print(' ')
print(classification_report(trainLabels, preds))

print('LSTM Classficiation Val Set')
preds = jointModel.predict_classes([testFeats, testFeats])
print(' ')
print(classification_report(testLabels, preds))

print('Counts Val UNBalanced')
getCounts(testLabels)
print('Classification MAX Score Val UNBalanced')
getClassificationScoreMaxCriteria(testFeats, testLabels)


jointModel.save('/home/jcleon/Storage/ssd0/ModelsLSTM/evenWithForwardK' + str(K) + '.h5') 
