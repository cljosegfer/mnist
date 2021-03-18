rm(list=ls())

pca = function(data, file){
  X = data.matrix(data[, -(1:2)])
  pca = prcomp(X)
  data.pca = pca$x
  path = paste('data/', file, '-pca.Rdata', sep = '')
  save(data.pca, file = path, compress = FALSE)
}

# read data
data = read.csv(file = 'dados/trainReduzido.csv')
valida = read.csv(file = 'dados/validacao.csv')

# pca
pca(data, file = 'mnist')
pca(valida, file = 'valida')

# # test
# load(file = 'data/mnist-pca.Rdata')
# load(file = 'data/valida-pca.Rdata')
