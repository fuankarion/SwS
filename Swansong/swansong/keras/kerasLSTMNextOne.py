# LSTM for international airline passengers problem with regression framing
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
import math
import matplotlib.pyplot as plt
from keras.layers import Activation
import numpy
import pandas
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import keras.backend.tensorflow_backend as K


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)
    
look_back = 5


# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
dataframe = pandas.read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')


# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
#Just start and end
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print('train ',train)
print('train.shape ',train.shape)

# reshape into X=t and Y=t+1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

print('trainX', trainX)
print('trainX.shape', trainX.shape)
print('trainY', trainY)
print('trainY.shape', trainY.shape)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

print('trainXReshaped.shape', trainX.shape)
print('trainY', trainY)


print('XXXData assemybly done, begintrain')
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(10, input_dim=look_back))
model.add(Activation('relu'))
#model.add(Dense(2, init='uniform'))
#model.add(Activation('relu'))
model.add(Dense(1))

model.summary()


model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=300, batch_size=4, verbose=2)


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert scale predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))



# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:,:] = numpy.nan
trainPredictPlot[look_back:len(trainPredict) + look_back,:] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:,:] = numpy.nan
testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset)-1,:] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()