using Flux, Images, ImageIO

# Carregamento do modelo treinado
modelo_carregado = Flux.load("modelo_treinado.bson")

# Função de reconhecimento
function reconhecer_imagem(modelo, imagem_path)
  # Carregamento e pré-processamento da nova imagem (mesmo código do exemplo anterior)
  # ...

  # Reconhecimento da imagem
  previsao = modelo(nova_imagem_array)

  # Retorno da previsão (probabilidades para cada classe)
  return previsao
end

# Exemplo de uso
imagem_path = "caminho/para/nova/imagem.jpg"
previsao = reconhecer_imagem(modelo_carregado, imagem_path)

# Impressão da previsão
println(previsao)

# Obtenção da classe com maior probabilidade
classe_prevista = argmax(previsao)
println("Classe prevista: $classe_prevista")




## Execute ##
# $ cd .\perceptronxor\src\
# $ julia cnn_reconh.jl