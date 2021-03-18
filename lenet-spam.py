# implementacao como em:
# https://medium.com/@mgazar/lenet-5-in-9-lines-of-code-using-keras-ac99294c8086

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import keras
import keras.layers as layers

from keras.preprocessing.image import ImageDataGenerator

# read data
mnist = pd.read_csv('dados/trainReduzido.csv')
dat_id = mnist[['Unnamed: 0']].to_numpy()
labels = mnist[['label']].to_numpy().reshape(-1)
mnist = mnist.drop(columns = ['Unnamed: 0', 'label']).to_numpy().reshape(len(mnist), 28, 28, 1)
# read validacao
valida = pd.read_csv('dados/validacao.csv')
val_id = valida[['Unnamed: 0']].to_numpy()
true = np.repeat([1, 5, 6, 7], repeats = 1000)
valida = valida.drop(columns = ['Unnamed: 0']).to_numpy().reshape(len(valida), 28, 28, 1)

# train, test, val sets
data = {}
data['features'], data['labels'] = mnist, labels
val = {}
val['features'], val['labels'] = valida, true
# test = {}
# train['features'], test['features'], train['labels'], test['labels'] = train_test_split(train['features'], train['labels'], test_size = 0.2, random_state = 0)

# pad w zeros, 28x28 -> 32x32
data['features'] = np.pad(data['features'], ((0,0), (2,2), (2,2), (0,0)), 'constant')
val['features'] = np.pad(val['features'], ((0,0), (2,2), (2,2), (0,0)), 'constant')

# k folds
k = 10
kfold = KFold(n_splits = k, shuffle = True)

# training param
EPOCHS = 10
BATCH_SIZE = 128

# metodo
idx = np.arange(len(data['labels']))
acuracia_best = 0
fold = 0
resultado = np.zeros(shape = [k, 1])
for idx_train, idx_test in kfold.split(idx):
    
    # train e test
    train = {}
    train['features'], train['labels'] = data['features'][idx_train], data['labels'][idx_train]
    test = {}
    test['features'], test['labels'] = data['features'][idx_test], data['labels'][idx_test]
    
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

    # X, Y
    X_train, y_train = train['features'], pd.get_dummies(train['labels'])
    X_test, y_test = test['features'], pd.get_dummies(test['labels'])
    train_generator = ImageDataGenerator().flow(X_train, y_train, batch_size = BATCH_SIZE)
    test_generator = ImageDataGenerator().flow(X_test, y_test, batch_size = BATCH_SIZE)
    
    # treinamento
    model.fit(train_generator, steps_per_epoch = X_train.shape[0] // BATCH_SIZE, epochs = EPOCHS, shuffle = True, verbose = 0)
    
    # inferencia e avaliacao
    score = model.evaluate(test_generator, verbose=0)
    
    # # unmute
    # model.fit(train_generator, steps_per_epoch = X_train.shape[0] // BATCH_SIZE, epochs = EPOCHS, shuffle = True)
    # score = model.evaluate(test_generator)
    
    # report
    acuracia = score[1]
    resultado[fold, 0] = acuracia
    fold += 1
    print(fold)
    
    # # best case
    # if acuracia > acuracia_best:
    #     acuracia_best = acuracia
    #     best = model
    
    # spam de submission
    # validacao
    score = model.evaluate(val['features'], pd.get_dummies(val['labels']))
    print('Test accuracy:', score[1])
    
    # submission
    predictions = np.argmax(model(val['features']).numpy(), axis = 1)
    predictions[predictions == 1] = 5
    predictions[predictions == 0] = 1
    predictions[predictions == 2] = 6
    predictions[predictions == 3] = 7
    submission = pd.concat([pd.DataFrame(val_id, columns = ['ImageId']), pd.DataFrame(predictions, columns = ['Label'])], axis = 1)
    submission.to_csv('submission/lenet5-parcial-' + 
                      str(score[1]) + '.csv', header = True, index = False)

# report
report = pd.DataFrame([np.mean(resultado), np.std(resultado)], 
                      columns = ['geral'], index = ['mean', 'sd'])
print(report)
