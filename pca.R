rm(list=ls())

pca = function(data, file){
  X = data.matrix(data[, -(1:2)])
  pca = prcomp(X)
  data.pca = pca$x
  path = paste('data/', file, '-pca.rds', sep = '')
  saveRDS(data.pca, file = path, compress = FALSE)
}

# read data
data = read.csv(file = 'dados/trainReduzido.csv')
valida = read.csv(file = 'dados/validacao.csv')

# # pca
# pca(data, file = 'mnist')
# pca(valida, file = 'valida')

# test
data.pca = readRDS(file = 'data/mnist-pca.rds')
valida.pca = readRDS(file = 'data/valida-pca.rds')
