rm(list=ls())

# svm
library(kernlab)

# view
MostraImagem = function(x, main = 'imagem', n = 28){
  rotate = function(x) t(apply(x, 2, rev))
  img = matrix(x, nrow = n)
  cor = rev(grey(50:1/50))
  image(rotate(img), col = cor, main = main)
}

matriz = function(x, n = 28){
  img = matrix(x, nrow = n)
  img = t(img)
  return(matrix(img, nrow = n * n))
}

# avaliacao
accuracy = function(y, yhat){
  erro = matrix(data = 0, nrow = length(y), ncol = 1)
  mistake = which(y != yhat)
  erro[mistake] = 1
  acuracia = 1 - sum(erro) / length(erro)
  return(acuracia)
}

# read data
data = read.csv(file = 'dados/trainReduzido.csv')
data.pca = readRDS(file = 'data/mnist-pca.rds')
m = 40
data = cbind(data[, 1:2], data.pca[, 1:m])

# # view
# i = 13414
# amostra = matriz(as.matrix(data[i, -(1:2)]))
# MostraImagem(amostra)

# param svm
type = 'C-bsvc'
kernel = 'rbfdot'
kpar = 'automatic'
C = 1

# k folds
k = 10
shuffle = sample(nrow(data))
data = data[shuffle, ]
folds = cut(seq(1, nrow(data)), breaks = k, labels = FALSE)

# metodo
resultado = matrix(data = 0, nrow = k, ncol = 5)
acuracia.best = 0
for(fold in seq(k)){
  # fold = 1
  print(fold)
  idx = which(folds == fold, arr.ind = TRUE)
  test = as.matrix(data[idx, -1])
  train = as.matrix(data[-idx, -1])

  # treinamento
  model = ksvm(x = train[, -1], y = train[, 1],
               type = type, kernel = kernel, kpar = kpar, C = C,
               scale = FALSE)

  # inferencia
  yhat = predict(object = model, newdata = test[, -1])

  # avaliacao
  y = test[, 1]
  acuracia = accuracy(y = y, yhat = yhat)

  # por classe
  c1 = which(y == 1)  # um
  acuracia.c1 = accuracy(y = y[c1], yhat = yhat[c1])
  c2 = which(y == 5)  # cinco
  acuracia.c2 = accuracy(y = y[c2], yhat = yhat[c2])
  c3 = which(y == 6)  # seis
  acuracia.c3 = accuracy(y = y[c3], yhat = yhat[c3])
  c4 = which(y == 7)  # sete
  acuracia.c4 = accuracy(y = y[c4], yhat = yhat[c4])

  # report
  resultado[fold, 1] = acuracia
  resultado[fold, 2] = acuracia.c1
  resultado[fold, 3] = acuracia.c2
  resultado[fold, 4] = acuracia.c3
  resultado[fold, 5] = acuracia.c4

  # best case
  if(acuracia > acuracia.best){
    acuracia.best = acuracia
    best = model
  }
}

# report
report = rbind(c(mean(resultado[, 1]), sd(resultado[, 1])),
               c(mean(resultado[, 2]), sd(resultado[, 2])),
               c(mean(resultado[, 3]), sd(resultado[, 3])),
               c(mean(resultado[, 4]), sd(resultado[, 4])),
               c(mean(resultado[, 5]), sd(resultado[, 5])))
rownames(report) = c('geral', 'c1', 'c2', 'c3', 'c4')
colnames(report) = c('mean', 'sd')
write.csv(report, file = 'report.csv')

# print
print(C)
print(report)

# validacao
valida.data = read.csv(file = 'dados/validacao.csv')
valida.pca = readRDS(file = 'data/valida-all-pca.rds')
valida = valida.pca[, 1:m]
predictions = predict(object = best, newdata = valida)
true = c(rep(1, 1000), rep(5, 1000), rep(6, 1000), rep(7, 1000))
acuracia.valida = accuracy(y = true, yhat = predictions)
print(acuracia.valida)

# submission
submission = cbind(valida.data[, 1], predictions)
colnames(submission) = c('ImageId', 'Label')
path = paste('submission/svm-pca-parcial-', acuracia.valida, '.csv', sep = '')
write.csv(submission, file = path, row.names = FALSE)
