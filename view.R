rm(list=ls())

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

# read data
data = read.csv(file = 'dados/trainReduzido.csv')
valida = read.csv(file = 'dados/validacao.csv')

# label, id
id.label = as.matrix(data[, 1:2])
id.valida = as.matrix(valida[, 1])

# clusters
cluster.1 = valida[1:1000, ]
cluster.2 = valida[1001:2000, ]
cluster.3 = valida[2001:3000, ]
cluster.4 = valida[3001:4000, ]

# # single
# i = 11746
# amostra = matriz(as.matrix(data[i, -(1:2)]))
# MostraImagem(amostra, main = paste('Imagem ', i))

# # varredura
# N = 100
# for(rep in seq(N)){
#   # random = floor(runif(n = 1, min = 1, max = 4000))
#   i = rep + 3852
#   amostra = matriz(as.matrix(valida[i, -1]))
#   MostraImagem(amostra, main = paste('imagem', i))
# }

# histograma
label = as.matrix(data[, 2])
cum.c1 = length(which(label == 1)) / length(label)
cum.c2 = length(which(label == 5)) / length(label)
cum.c3 = length(which(label == 6)) / length(label)
cum.c4 = length(which(label == 7)) / length(label)
cum = cbind(cum.c1, cum.c2, cum.c3, cum.c4)
# hist(label)
barplot(cum, main = 'MNIST', xlab = 'Dígitos', 
        names.arg=c('1', '5', '6', '7'))
