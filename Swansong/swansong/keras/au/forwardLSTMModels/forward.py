from keras.models import load_model
import sys
sys.path.append('../')
from trainCore import *

import matlab.engine
import matlab

aK = 40
au = 'AU01'
view = 'v1'
jointModel = load_model('/home/jcleon/Storage/ssd0/ModelsLSTM/'+ '/' + au + '-' + view + '-K' + str(aK) + '.h5')


txtFeatsTestDir = '/home/jcleon/Storage/disk2/foldFeats/Val/'+au+'_'+view+'_Fold1' #always fold1
print('txtFeatsTestDir',txtFeatsTestDir)

bulkLoadDirVal = '/home/jcleon/Storage/ssd0/BL/Val/'
print('bulkLoadDirVal',bulkLoadDirVal)
testSubjects = ['F007', 'F008', 'F009', 'F010', 'F011', 'M001', 'M002', 'M003', 'M004', 'M005', 'M006', 'rF001', 'rF002', 'rM001', 'rM002', 'rM003', 'rM004', 'rM005', 'rM006', 'rM007']
tasksTest = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'T13', 'T14', 'T15']
testFeats, testLabels = loadSet(testSubjects, tasksTest, txtFeatsTestDir, view, aK, bulkLoadDirVal, au)

print('txtFeatsTestDir',txtFeatsTestDir)
print('bulkLoadDirVal',bulkLoadDirVal)
print('LSTM Classficiation Val Set')
print('testFeats',testFeats);
probs = jointModel.predict_proba([testFeats, testFeats])

eng = matlab.engine.start_matlab()
probsAsArray = probs.tolist()
testLabelsAsArray = testLabels.tolist()


print('Counts Val UNBalanced')
getCounts(testLabels)
print('Classification MAX Score Val UNBalanced')
predsMax,cr = getClassificationScoreMaxCriteria(testFeats, testLabels)


#print('probsAsArray ',probsAsArray)
#print('testLabelsAsArray ',testLabelsAsArray)

ps = eng.CalcRankPerformance(matlab.int8(testLabelsAsArray), matlab.double(probsAsArray), 1, 'All')
F1_MAX_LSTM = max(np.array(ps['F1']))[0]
print('F1_MAX_LSTM ', F1_MAX_LSTM)

ps = eng.CalcRankPerformance(matlab.int8(testLabelsAsArray), matlab.double(predsMax), 1, 'All')
F1_MAX_PRED_MAX = max(np.array(ps['F1']))[0]
print('F1_MAX_PRED_MAX ', F1_MAX_PRED_MAX)
