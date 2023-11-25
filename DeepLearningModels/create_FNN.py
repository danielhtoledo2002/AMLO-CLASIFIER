# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 01:52:57 2023

@author: alexi
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import text_pro as text
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import seaborn as sns
import joblib


#import dataset
df = pd.read_csv("amlo_clasify_chatpgt3.csv")


# check if gpu is working
if tf.test.gpu_device_name():
  print(f'se encontró el siguiente gpu: {tf.test.gpu_device_name()}')
else:
  print('aqui no hay gpu')
  
# string to a vector 
df['vector'] = df["vector"].apply(lambda x: np.fromstring( x.replace('\n','') .replace('[','') .replace(']','') .replace(' ',' '), sep=' '))

#Since the model uses numbers, we are going to change the classifications names to numbers

df['clas_num'] = df['classification_spanish'].apply(text.clasification_to_num)

# we create the data to train and test the model
X = df['vector']
X = np.concatenate(X, axis= 0).reshape(-1, 300)
y =  df['clas_num']

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.80, random_state=101, stratify=y)


#
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#create an early stopping
early_stopping = EarlyStopping(monitor='val_loss',
                               patience=6,
                               restore_best_weights=True)


# transform data to fit the model and its classifications
y_train = to_categorical(y_train, 10) 
y_test = to_categorical(y_test, 10)

#the "skeleton" of the model is initialized
model = Sequential()

#we add layers to the model
model.add(Dense(units=45, activation='relu', input_dim=300))
model.add(Dropout(0.2))
model.add(Dense(units=40, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=10, activation='softmax'))

#compile the model
model.compile(optimizer= Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

#train model
history = model.fit(X_train,
                    y_train,
                    batch_size=20,
                    epochs=200,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping])



metrics = pd.DataFrame(history.history)


y_pred = np.argmax(model.predict(X_test), axis=-1)
y_test = np.argmax(y_test, axis=-1)

print(classification_report(y_test, y_pred))

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
sns.lineplot(ax=axes[0], data=metrics[['loss', 'val_loss']])
sns.lineplot(ax=axes[1],data=metrics[['accuracy', 'val_accuracy']])

sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True,
            fmt='.0f', ax=axes[2])

model.save('FNN_model2.keras')
joblib.dump(scaler, 'scaler2.save')



