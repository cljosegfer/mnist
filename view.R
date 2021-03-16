rm(list=ls())

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

N = 100
for(rep in seq(N)){
  # random = floor(runif(n = 1, min = 1, max = 4000))
  i = rep + 3852
  amostra = matriz(as.matrix(valida[i, -1]))
  MostraImagem(amostra, main = paste('imagem', i))
}
