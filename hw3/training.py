import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Input
from keras.layers import Conv2D
from keras.utils import np_utils

def load_data(num=5000):
    train = pd.read_csv('./train.csv')
    x_train = np.array( [ row.split(' ') for row in train['feature'].tolist() ] )
    y_train = train['label'].tolist()
    print('total train shape: ', x_train.shape)
    # return train & validation data
    return (x_train[0:num], y_train[0:num]),(x_train[num:num+500], y_train[num:num+500])

def normalize_255(x):
    x = x/255
    return x

def classify(y):
    np_utils.to_categorical(y, 7)

def train():
    print('---training started---')
    # 1. load data and preProcess
    (x_train, y_train), (x_validation, y_validation) = load_data()
    x_train = normalize_255(x_train)
    x_validation = normalize_255(x_validation)
    y_train = classify(y_train)
    y_validation = classify(y_validation)
    print('y_validation shape: ', y_validation.shape, 'y_validation[0]', y_validation[0])
    # 2. build model - cnn
    model = Sequential()

    inputs = Input( shape=())
    conv_1 = Conv2D(1, (3,3), strides=(1,1), padding='same')(inputs)
    act_1 = Activation('relu')(conv_1)

    # model = Model(inputs=)

    # 3. training

    # 4. save mode

train()