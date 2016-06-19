import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt


ds = pd.read_csv('../dataset/breast-cancer-wisconsin.data.txt', sep=",", header=None)

ds.columns = ['id', 'ClumpT', 'CellSize', 'CellShape', 'MarAdh', 'ECellSize', 'BareNuclei', 'Chromatin', 'NNucleoli', 'Mitoses', 'Class']

ds.replace('?', np.nan, inplace=True)
ds.dropna(inplace=True, how='any')

ds['Class'].replace(2, 0, inplace=True)
ds['Class'].replace(4, 1, inplace=True)


ds = ds[['ClumpT', 'CellSize', 'CellShape', 'MarAdh', 'ECellSize', 'BareNuclei', 'Chromatin', 'NNucleoli', 'Mitoses', 'Class']]



def normalize(column):
    return (column - column.mean()) / (column.max() - column.min())


ClumpT_norm = normalize(pd.to_numeric(ds['ClumpT']))
CellSize_norm = normalize(pd.to_numeric(ds['CellSize']))
CellShape_norm = normalize(pd.to_numeric(ds['CellShape']))
MarAdh_norm = normalize(pd.to_numeric(ds['MarAdh']))
ECellSize_norm = normalize(pd.to_numeric(ds['ECellSize']))
BareNuclei_norm = normalize(pd.to_numeric(ds['BareNuclei']))
Chromatin_norm = normalize(pd.to_numeric(ds['Chromatin']))
NNucleoli_norm = normalize(pd.to_numeric(ds['NNucleoli']))
Mitoses_norm = normalize(pd.to_numeric(ds['Mitoses']))



X_train = np.asarray(pd.concat([ClumpT_norm, CellSize_norm, CellShape_norm, MarAdh_norm, ECellSize_norm, BareNuclei_norm, Chromatin_norm, NNucleoli_norm, Mitoses_norm], axis=1, join='inner'))
Y_train = np.asarray(pd.to_numeric(ds['Class']))



model = Sequential()

model.add(Dense(output_dim=15, input_dim=9))
model.add(Activation("relu"))
model.add(Dense(output_dim=1))
model.add(Activation("linear"))

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

filepath = '../checkpoints/weights-cancer.best.hdf5'

checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit(X_train[:400], Y_train[:400], callbacks=callbacks_list, nb_epoch=20, batch_size=32, verbose=0)

loss_history = history.history['loss']
acc_history = history.history['acc']
epochs = [(x + 1) for x in range(20)]




X_test = X_train[400:]
Y_test = Y_train[400:]

loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)
print loss_and_metrics

ax = plt.subplot(211)
ax.plot(epochs, loss_history, color='red')
ax.set_xlabel('Epochs')
ax.set_ylabel('Error Rate\n')
ax.set_title('Error Rate for Epoch\n')

ax2 = plt.subplot(212)
ax2.plot(epochs, acc_history, color='c')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy\n')
ax2.set_title('Accuracy for Epoch\n')

plt.subplots_adjust(hspace=0.8)

plt.show()