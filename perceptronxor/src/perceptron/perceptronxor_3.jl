using Random
using Printf

# Definição da estrutura da Rede Neural (MLP)
mutable struct MLP
    input_size::Int
    hidden_size::Int
    output_size::Int
    weights_input_hidden::Matrix{Float64}  # Pesos da camada de entrada para oculta
    weights_hidden_output::Matrix{Float64} # Pesos da camada oculta para saída
    bias_hidden::Vector{Float64}           # Bias da camada oculta
    bias_output::Float64                   # Bias da camada de saída
    learning_rate::Float64                 # Taxa de aprendizado
end

# Construtor da rede com inicialização otimizada
function MLP(input_size::Int, hidden_size::Int, output_size::Int, learning_rate::Float64)
    # Inicialização He-et-al para pesos (melhor para redes profundas)
    weights_input_hidden = randn(input_size, hidden_size) * sqrt(2.0/input_size)
    weights_hidden_output = randn(hidden_size, output_size) * sqrt(2.0/hidden_size)

    # Biases inicializados com zeros para evitar saturação inicial
    bias_hidden = zeros(hidden_size)
    bias_output = 0.0

    MLP(input_size, hidden_size, output_size,
        weights_input_hidden, weights_hidden_output,
        bias_hidden, bias_output, learning_rate)
end

# Função de ativação sigmoid e sua derivada
sigmoid(x) = 1.0 ./ (1.0 .+ exp.(-x))
sigmoid_derivative(x) = x .* (1.0 .- x)

# Função de forward pass
function forward(mlp::MLP, inputs::Vector{Float64})
    # Cálculo da camada oculta
    hidden_inputs = mlp.weights_input_hidden' * inputs .+ mlp.bias_hidden
    hidden_outputs = sigmoid.(hidden_inputs)

    # Cálculo da saída final
    output_input = mlp.weights_hidden_output' * hidden_outputs .+ mlp.bias_output
    output = sigmoid(output_input)

    return hidden_outputs, output[1]  # Retorna ativações e saída final
end

# Função de treinamento com backpropagation
function train!(mlp::MLP, inputs::Vector{Float64}, target::Float64)
    # === Forward pass ===
    hidden_outputs, output = forward(mlp, inputs)

    # === Backpropagation ===
    # Erro na camada de saída (delta_o)
    output_error = (output - target) * output * (1 - output)

    # Erro na camada oculta (delta_h)
    hidden_errors = mlp.weights_hidden_output .* output_error .* hidden_outputs .* (1 .- hidden_outputs)

    # === Atualização de pesos ===
    # Atualiza pesos da camada oculta->saída
    Δweights_hidden_output = mlp.learning_rate * output_error * hidden_outputs
    mlp.weights_hidden_output .-= Δweights_hidden_output

    # Atualiza bias da saída
    mlp.bias_output -= mlp.learning_rate * output_error

    # Atualiza pesos da camada entrada->oculta (transpõe inputs para multiplicação correta)
    Δweights_input_hidden = mlp.learning_rate * inputs * hidden_errors'
    mlp.weights_input_hidden .-= Δweights_input_hidden

    # Atualiza bias da camada oculta
    mlp.bias_hidden .-= mlp.learning_rate * vec(hidden_errors)
end

# Função principal de execução
function main()
    Random.seed!(123)  # Fixa seed para reproducibilidade
    mlp = MLP(2, 4, 1, 0.05)  # Cria rede com 2 entradas, 4 neurônios ocultos, 1 saída
    training_data = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ]

    # === Treinamento ===
    for epoch in 1:50000
        # Treina com todos os dados de treinamento
        for (inputs, target) in training_data
            train!(mlp, inputs, target)
        end

        # === Verificação de convergência a cada 1000 épocas ===
        if epoch % 1000 == 0
            total_loss = 0.0
            for (inputs, target) in training_data
                _, output = forward(mlp, inputs)
                total_loss += (output - target)^2  # Erro quadrático
            end
            @printf("Epoch: %5d | Loss: %.4f\n", epoch, total_loss)
        end
    end

    # === Teste final ===
    @printf("\n\n================ TESTE FINAL ===============\n")
    threshold = 0.5
    resposta = ""

    for (inputs, target) in training_data
        _, output = forward(mlp, inputs)
        classification = output ≥ threshold ? 1 : 0
        linha = @sprintf("Entradas: %-10s | Alvo: %-2d | Saída: %-6.4f | Classificação: %d",
                         inputs, target, output, classification)
        resposta *= linha * "\n"
        println("\n----------------------------------------")
        println(linha)
    end

    println("\n\n======= RESULTADO FINAL =======")
    println(resposta)
end

main()

## Execute ##
# $ cd .\perceptronxor\src\
# $ julia perceptronxor_3.jl
