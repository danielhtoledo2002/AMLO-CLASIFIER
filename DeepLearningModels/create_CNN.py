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
df = pd.read_csv("OpenAi/amlo_clasify_chatpgt3.csv")


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

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.70, random_state=101, stratify=y)


#
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#create an early stopping
early_stopping = EarlyStopping(monitor='val_loss',
                               patience=5,
                               restore_best_weights=True)


# transform data to fit the model and its classifications
y_train = to_categorical(y_train, 10) 
y_test = to_categorical(y_test, 10)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1],1)


#create model
model = Sequential()
model.add(Conv1D(46,2, activation='relu', input_shape=X_train[0].shape))
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10,activation='softmax'))

#compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

#train model
history = model.fit(X_train,
                    y_train,
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

model.save('CNN_model.keras')
joblib.dump(scaler, 'scaler.save')



