using CSV
using DataFrames
using Flux
using Random

# Definindo a função sigmoide
function sigmoid(x)
    return 1.0 ./ (1.0 .+ exp.(-x))
end

# Função para solicitar entradas do usuário
function obter_entradas()
    println("Insira as características de saúde:")
    println("1. Peso (kg)")
    peso = parse(Float32, readline())
    println("2. Altura (m)")
    altura = parse(Float32, readline())
    println("3. Idade (anos)")
    idade = parse(Float32, readline())
    println("4. Nível de atividade física (1-5)")
    atividade_fisica = parse(Float32, readline())
    return [peso, altura, idade, atividade_fisica]
end

# Função para interpretar as saídas
function interpretar_saida(saidas)
    imc = saidas[1]
    atividade1 = saidas[2]
    atividade2 = saidas[3]

    imc_msg = imc < 0.5 ? "IMC baixo. Pode ser necessário ganhar peso para uma saúde ideal." :
                         imc < 0.75 ? "IMC normal. Mantenha um estilo de vida saudável." :
                                      "IMC alto. Pode ser necessário perder peso para uma saúde ideal."

    atividade_msg1 = atividade1 < 0.5 ? "Nível baixo de atividade física recomendado." :
                                       atividade1 < 0.75 ? "Nível moderado de atividade física recomendado." :
                                                        "Nível alto de atividade física recomendado."
    
    atividade_msg2 = atividade2 < 0.5 ? "Considere atividades físicas leves como caminhada." :
                                       atividade2 < 0.75 ? "Considere atividades físicas moderadas como ciclismo." :
                                                        "Considere atividades físicas intensas como corrida."

    return (imc_msg, atividade_msg1, atividade_msg2)
end

# Carregar dados de treinamento (substitua pelo seu arquivo CSV)
dados = CSV.read("dados_imc.csv", DataFrame)
dados.peso = convert.(Float32, dados.peso)
dados.altura = convert.(Float32, dados.altura)
dados.idade = convert.(Float32, dados.idade)
dados.atividade_fisica = convert.(Float32, dados.atividade_fisica)
dados.imc = convert.(Float32, dados.imc)

# Preparar os dados
X = Matrix(dados[:, [:peso, :altura, :idade, :atividade_fisica]])
y = dados.imc

# Inicializar pesos e biases aleatoriamente
Random.seed!(0) # Para garantir a reprodutibilidade
W_input_hidden = randn(Float32, 5, 4)
b_hidden = randn(Float32, 5)
W_hidden_output = randn(Float32, 3, 5)
b_output = randn(Float32, 3)

# Taxa de aprendizado e número de épocas
lr = 0.01
epochs = 1000

# Treinamento manual da rede neural
function train_network(W_input_hidden, b_hidden, W_hidden_output, b_output, X, y, epochs, lr)
    for epoch in 1:epochs
        # Forward pass
        hidden_input = W_input_hidden * X' .+ b_hidden
        hidden_output = sigmoid.(hidden_input)

        output_input = W_hidden_output * hidden_output .+ b_output
        output_output = sigmoid.(output_input)

        # Calculando a perda
        loss_value = mse(output_output, y')

        # Backpropagation
        gradients = Flux.gradient(() -> loss_value, (W_input_hidden, b_hidden, W_hidden_output, b_output))

        # Atualizando os pesos e biases
        W_input_hidden .-= lr * gradients[1]
        b_hidden .-= lr * gradients[2]
        W_hidden_output .-= lr * gradients[3]
        b_output .-= lr * gradients[4]

        if epoch % 100 == 0
            println("Epoch: $epoch, Loss: $(loss_value.data)")
        end
    end
    return W_input_hidden, b_hidden, W_hidden_output, b_output
end

# Treinar a rede neural
W_input_hidden, b_hidden, W_hidden_output, b_output = train_network(W_input_hidden, b_hidden, W_hidden_output, b_output, X, y, epochs, lr)

# Loop interativo para entrada de dados e cálculo de IMC
while true
    print("Digite o peso (kg) ou 'sair' para encerrar: ")
    entrada_peso = readline()

    if entrada_peso == "sair"
        break
    end

    try
        x = obter_entradas()

        # Forward pass para a entrada do usuário
        hidden_input = W_input_hidden * x .+ b_hidden
        hidden_output = sigmoid.(hidden_input)

        output_input = W_hidden_output * hidden_output .+ b_output
        output_output = sigmoid.(output_input)

        # Imprimir as ativações da camada oculta e saídas
        println("Ativações na Camada Oculta: ", hidden_output)
        println("Saídas: ", output_output)

        # Interpretar as saídas
        (imc_msg, recomendacao_msg1, recomendacao_msg2) = interpretar_saida(output_output)
        println("---------------------")
        println("IMC estimado: ", output_output[1], " - ", imc_msg)
        println("---------------------")
        println("Recomendações de atividade física:")
        println("1. ", recomendacao_msg1)
        println("2. ", recomendacao_msg2)

    catch e
        println("Entrada inválida. Certifique-se de digitar números válidos para todas as características.")
    end
end



## Execute ##
# $ julia imc2.jl