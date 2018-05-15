from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape)

# image = x_train[0]
# plt.imshow(image, 'gray')
# plt.show()

# cv.imshow(image)
# cv.waitKey(0)
# cv.destroyAllWindows()

model = Sequential()

model.add(Dense(input_dim=28*28, units=500, activation='relu'))
model.add(Dense(units=500, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
optimizer='adam',
metrics=['accuracy'])

xto_train=x_train[:5000,]
yto_train=y_train[:5000,]

model.fit(xto_train,yto_train, batch_size=1000, epochs=20)

score=model.evaluate(x_test[:500], y_test[:500])
print('Total loss:', score[0])
print('Accuracy:', score[1])