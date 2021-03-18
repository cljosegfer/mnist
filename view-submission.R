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
file = 'lenet5-parcial-0.9952499866485596.csv'
path = paste('submission/', file, sep = '')
submission = read.csv(file = path)

# encontra os erros
true = c(rep(1, 1000), rep(5, 1000), rep(6, 1000), rep(7, 1000))
errado = which(true != submission[, 2])

# # single
# i = 91
# amostra = matriz(as.matrix(valida[i, -1]))
# MostraImagem(amostra, main = paste('Imagem', i, 'modelo previu', submission[i, 2]))


# varredura
for(i in errado){
  amostra = matriz(as.matrix(valida[i, -1]))
  MostraImagem(amostra, main = paste('Imagem', i, 
                                     'modelo previu', submission[i, 2], 
                                     'mas isso é', true[i]))
}
