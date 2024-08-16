
using StatsBase 

# Continuando com o conjunto de dados numéricos
dados_numéricos = ['1', '2', '2', '3', '3', '3', '4', '4', '4', '4']

# Encontre todas as modas
modas = modes(dados_numéricos)

println("As modas do conjunto de dados são: ", modas)

dados_array = [1,6, 9, 8, 2, 3, 4, 5,1, 2, 3,1,0]
sort!(dados_array)
println(dados_array) 
mediana = median(dados_array)
println("A mediana do conjunto de dados é: ", mediana)