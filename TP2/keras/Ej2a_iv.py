'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

import matplotlib.pyplot as plt

batch_size = 128
nb_classes = 10
nb_epoch = 20

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

dpvs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for dp in dpvs:
	model = Sequential()
	model.add(Dense(512, input_shape=(784,)))
	model.add(Activation('relu'))
	model.add(Dropout(dp))
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(dp))
	model.add(Dense(10))
	model.add(Activation('softmax'))

	model.summary()

	model.compile(loss='categorical_crossentropy',
		      optimizer=RMSprop(),
		      metrics=['accuracy'])

	history = model.fit(X_train, Y_train,
		            batch_size=batch_size, nb_epoch=nb_epoch,
		            verbose=0, validation_data=(X_test, Y_test))
	score = model.evaluate(X_test, Y_test, verbose=0)
	#print('Test score:', score[0])
	print(dp)
	print('Test accuracy:', score[1])

	axes = plt.gca()
	axes.set_ylim([0.7,1])
	plt.plot(history.history['acc'],color='r')
	plt.plot(history.history['val_acc'],color='b')
	plt.title('Dropout = %.0f%%' % (dp*100))
	plt.ylabel('Accuracy')
	plt.xlabel('Épocas')
	plt.legend(['Train', 'Test'], loc='lower right')
	plt.show()

