from keras.layers import Activation
from keras.layers import Dense
from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# for a single-input model with 2 classes (binary):
model = Sequential()

#model.add(Dense(1, input_dim=784, activation='sigmoid'))

model = Sequential()
model.add(Dense(2, input_dim=784,init='uniform'))
model.add(Activation('sigmoid'))
model.add(Dense(2, init='uniform'))
model.add(Activation('sigmoid'))
#model.add(Dense(2, init='uniform'))
#model.add(Activation('sigmoid'))
model.add(Dense(1, init='uniform'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# generate dummy data
import numpy as np
data = np.random.random((1000, 784))

print('data.shape ',data.shape)
sums = np.sum(data, axis=1)
print('sums.shape ',sums.shape)

labels=np.zeros(sums.shape)
meanSums=np.mean(sums)
labels[sums < meanSums] = 0
labels[sums > meanSums] = 1

print(labels)


# train the model, iterating on the data in batches
# of 32 samples
model.fit(data, labels, nb_epoch=200, batch_size=32)