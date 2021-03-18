# implementacao como em:
# https://medium.com/@mgazar/lenet-5-in-9-lines-of-code-using-keras-ac99294c8086

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import keras
import keras.layers as layers

from keras.preprocessing.image import ImageDataGenerator

# read data
data = pd.read_csv('dados/trainReduzido.csv')
dat_id = data[['Unnamed: 0']].to_numpy()
labels = data[['label']].to_numpy().reshape(-1)
data = data.drop(columns = ['Unnamed: 0', 'label']).to_numpy().reshape(len(data), 28, 28, 1)
# read validacao
valida = pd.read_csv('dados/validacao.csv')
val_id = valida[['Unnamed: 0']].to_numpy()
true = np.repeat([1, 5, 6, 7], repeats = 1000)
valida = valida.drop(columns = ['Unnamed: 0']).to_numpy().reshape(len(valida), 28, 28, 1)

# train, set, val
train = {}
val = {}
train['features'], train['labels'] = data, labels
val['features'], val['labels'] = valida, true
test = {}
train['features'], test['features'], train['labels'], test['labels'] = train_test_split(train['features'], train['labels'], test_size = 0.2, random_state = 0)

# pad w zeros, 28x28 -> 32x32
train['features'] = np.pad(train['features'], ((0,0), (2,2), (2,2), (0,0)), 'constant')
test['features'] = np.pad(test['features'], ((0,0), (2,2), (2,2), (0,0)), 'constant')
val['features'] = np.pad(val['features'], ((0,0), (2,2), (2,2), (0,0)), 'constant')

# model
model = keras.Sequential()
model.add(layers.Conv2D(filters = 6, kernel_size = (3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(layers.AveragePooling2D())
model.add(layers.Conv2D(filters = 16, kernel_size = (3, 3), activation = 'relu'))
model.add(layers.AveragePooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(units = 120, activation= 'relu'))
model.add(layers.Dense(units = 84, activation= 'relu'))
model.add(layers.Dense(units = 4, activation = 'softmax'))
model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adam(), metrics = ['accuracy'])

# training
EPOCHS = 10
BATCH_SIZE = 128

X_train, y_train = train['features'], pd.get_dummies(train['labels'])
X_test, y_test = test['features'], pd.get_dummies(test['labels'])

train_generator = ImageDataGenerator().flow(X_train, y_train, batch_size = BATCH_SIZE)
test_generator = ImageDataGenerator().flow(X_test, y_test, batch_size = BATCH_SIZE)

steps_per_epoch = X_train.shape[0] // BATCH_SIZE
test_steps = X_test.shape[0] // BATCH_SIZE

model.fit_generator(train_generator, steps_per_epoch = steps_per_epoch, epochs = EPOCHS, 
                    validation_data = test_generator, validation_steps = test_steps, shuffle = True)

# avaliacao
score = model.evaluate(val['features'], pd.get_dummies(val['labels']))
print('Test loss:', score[0])
print('Test accuracy:', score[1])
