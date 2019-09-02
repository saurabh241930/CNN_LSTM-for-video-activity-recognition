from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
import time
import os
import numpy as np
from keras.layers import Dense, Flatten, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import sys

X = np.load('X_data.npy')
y = np.load('y_data.npy')

X = np.squeeze(X)
y = np.squeeze(y)

model = Sequential()
model.add(LSTM(2048, return_sequences=False,input_shape=X[0].shape,dropout=0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y[0]), activation='softmax'))

early_stopper = EarlyStopping(patience=5)


adam_optimizer = Adam(lr=1e-5, decay=1e-6)

model.compile(optimizer=adam_optimizer, loss='categorical_crossentropy', metrics=['accuracy']
              
model.fit(X,y,batch_size=32,validation_split=0.01,verbose=1,callbacks=[early_stopper],epochs=1000)