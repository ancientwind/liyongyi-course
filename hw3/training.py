import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Input
from keras.layers import Conv2D

def load_data(num=5000):
    train = pd.read_csv('./train.csv')
    x_train = np.array( [ row.split(' ') for row in train['feature'].tolist()], dtype='float32')
    y_train = train['label'].tolist()
    print('total train shape: ', x_train.shape)
    return (x_train[0:num], y_train)

def train():
    # 1. load data
    (x_train, y_train) = load_data()
    # 2. build model - cnn
    model = Sequential()
    
    inputs = Input( shape=())
    conv_1 = Conv2D(1, (3,3), strides=(1,1), padding='same')(inputs)
    act_1 = Activation('relu')(conv_1)

    model = Model(inputs=)

    # 3. training

    # 4. save mode

if __name__ is '__main__':
    train()