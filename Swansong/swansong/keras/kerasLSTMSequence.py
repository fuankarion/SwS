# LSTM for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing import sequence
from keras.layers import Dropout
from keras.layers.convolutional import Convolution1D

embedding_vecor_length = 64
max_review_length = 2048
top_words = 5000

# fix random seed for reproducibility
numpy.random.seed(7)


# load the dataset but only keep the top n words, zero the rest
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)

X_test=X_test[:7500]
y_test=y_test[:7500]

print('X_train.shap',X_train.shape)
print('len(X_train[1000])',len(X_train[1000]))
print('y_train.shap',y_train.shape)
print('X_test.shap',X_test.shape)
print('y_test.shap',y_test.shape)

# truncate and pad input sequences
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

#X_train.reshape(X_train.shape + (1,))
#X_test.reshape(X_test.shape + (1,))


print('top_words ',top_words)
print('embedding_vecor_length ',embedding_vecor_length)
print('max_review_length ',max_review_length)

# create the model
"""
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Convolution1D(nb_filter=64, filter_length=3, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(LSTM(10))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
"""

model = Sequential()
model.add(Convolution1D(nb_filter=64, filter_length=3, border_mode='same',input_shape=(1, max_review_length), activation='relu'))
model.add(LSTM(10))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, nb_epoch=8, batch_size=25, verbose=1)

print('XXXX Fitting done forward')
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))