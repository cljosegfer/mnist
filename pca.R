rm(list=ls())

pca = function(data, file){
  X = data.matrix(data[, -1])
  pca = prcomp(X)
  data.pca = pca$x
  path = paste('data/', file, '-pca.rds', sep = '')
  saveRDS(data.pca, file = path, compress = FALSE)
}

# read data
data = read.csv(file = 'dados/trainReduzido.csv')
label = data[, 2]
data = data[, -2] # remover a coluna label q n existe em valida
valida = read.csv(file = 'dados/validacao.csv')

# # pca
# pca(data, file = 'mnist')
# pca(valida, file = 'valida')

# test
data.pca = readRDS(file = 'data/mnist-pca.rds')
valida.pca = readRDS(file = 'data/valida-pca.rds')

# plot
true = c(rep(1, 1000), rep(5, 1000), rep(6, 1000), rep(7, 1000))
plot(data.pca[, 1:2], col = label, main = 'Subconjunto MNIST')
plot(valida.pca[, 1:2], col = true, main = 'Conjunto de Validação')

# testando hipoteses
# all = rbind(data, valida)
# X = data.matrix(all[, -1])
# pca = prcomp(X)
# all.pca = pca$x
# idx.data = seq(13017)
# plot(all.pca[idx.data, 1:2], col = label, main = 'Subconjunto MNIST')
# plot(all.pca[-idx.data, 1:2], col = true, main = 'Conjunto de Validação')
# plot(all.pca[, 1:2], col = c(label, true), main = 'all')

# # novo pca para valida
# file = 'valida-all'
# path = paste('data/', file, '-pca.rds', sep = '')
# saveRDS(all.pca[-idx.data, ], file = path, compress = FALSE)
