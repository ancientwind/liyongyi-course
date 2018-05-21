from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def load_data(number=10000):
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train[0:number]
    y_train = y_train[0:number]

    # 3 dim -> 2 dim
    x_train = x_train.reshape(number, 28*28)
    x_test = x_test.reshape(x_test.shape[0], 28*28) 

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # number -> matrix representation of the number
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    print(x_train.shape, y_train[0])

    # nomailize to 0~1
    x_train = x_train / 255
    x_test = x_test / 255

    print(x_train[0])
    return (x_train, y_train), (x_test, y_test)

# image = x_train[0]
# plt.imshow(image, 'gray')
# plt.show()

# cv.imshow(image)
# cv.waitKey(0)
# cv.destroyAllWindows()

model = Sequential()

# Dense means fully connected layer
# units is the number of the neuron
model.add(Dense(input_dim=28*28, units=700, activation='relu'))
model.add(Dense(units=700, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
optimizer='adam',
metrics=['accuracy'])

(x_train, y_train), (x_test, y_test) = load_data(5000)

xto_train = x_train[0:4500]
xto_validation = x_train[4500:5000]
yto_train = y_train[0:4500]
yto_validation = y_train[4500:5000]

model.fit(xto_train, yto_train, validation_data=(xto_validation, yto_validation), batch_size=100, epochs=20)

# score=model.evaluate(xto_validation, yto_validation)
# print('Total loss:', score[0])
# print('Accuracy:', score[1])

result=model.evaluate(x_test[:500], y_test[:500])
print('Total loss:', result[0])
print('Accuracy:', result[1])

classes = model.predict(x_test[:100])
print(classes[:3])
print('origin: ', y_test[0])
print(np.argmax(classes[0]))

print('origin: ', y_test[1])
print(np.argmax(classes[1]))

print('origin: ', y_test[97])
print(np.argmax(classes[97]))

model.save_weights('model/keras_seq.h5')