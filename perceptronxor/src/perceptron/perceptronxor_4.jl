using Random
using Printf

# Definição da rede neural com múltiplas saídas
mutable struct MLP
    input_size::Int
    hidden_size::Int
    output_size::Int
    weights_input_hidden::Matrix{Float64}
    weights_hidden_output::Matrix{Float64}
    bias_hidden::Vector{Float64}
    bias_output::Vector{Float64}
    learning_rate::Float64
end

# Função de inicialização da rede
function MLP(input_size::Int, hidden_size::Int, output_size::Int, learning_rate::Float64)
    weights_input_hidden = randn(input_size, hidden_size) * sqrt(2.0/input_size)
    weights_hidden_output = randn(hidden_size, output_size) * sqrt(2.0/hidden_size)
    bias_hidden = zeros(hidden_size)
    bias_output = zeros(output_size)

    MLP(input_size, hidden_size, output_size,
        weights_input_hidden, weights_hidden_output,
        bias_hidden, bias_output, learning_rate)
end

# Funções de ativação
sigmoid(x) = 1.0 ./ (1.0 .+ exp.(-x))
sigmoid_derivative(x) = x .* (1.0 .- x)
softmax(x) = exp.(x) ./ sum(exp.(x))

# Função forward
function forward(mlp::MLP, inputs::Vector{Float64})
    hidden_inputs = mlp.weights_input_hidden' * inputs .+ mlp.bias_hidden
    hidden_outputs = sigmoid.(hidden_inputs)
    output_inputs = mlp.weights_hidden_output' * hidden_outputs .+ mlp.bias_output
    output = softmax(output_inputs)
    return hidden_outputs, output
end

# Função de treinamento
function train!(mlp::MLP, inputs::Vector{Float64}, target::Vector{Float64})
    hidden_outputs, output = forward(mlp, inputs)
    output_errors = output .- target
    hidden_errors = mlp.weights_hidden_output * output_errors .* sigmoid_derivative(hidden_outputs)

    mlp.weights_hidden_output .-= mlp.learning_rate .* (hidden_outputs * output_errors')
    mlp.bias_output .-= mlp.learning_rate .* output_errors
    mlp.weights_input_hidden .-= mlp.learning_rate .* (inputs * hidden_errors')
    mlp.bias_hidden .-= mlp.learning_rate .* hidden_errors
end

function main()
    Random.seed!(42)

    training_data = [
        ([255, 0, 127], "Rose"),
        ([127, 0, 0], "Vermelho"),
        ([255, 0, 0], "Vermelho"),
        ([255, 127, 0], "Laranja"),
        ([127, 127, 0], "Amarelo"),
        ([255, 255, 0], "Amarelo"),
        ([127, 255, 0], "Primavera"),
        ([0, 127, 0], "Verde"),
        ([0, 255, 0], "Verde"),
        ([0, 255, 127], "Turquesa"),
        ([0, 127, 127], "Ciano"),
        ([0, 255, 255], "Ciano"),
        ([0, 127, 255], "Cobalto"),
        ([0, 0, 127], "Azul"),
        ([0, 0, 255], "Azul"),
        ([127, 0, 255], "Violeta"),
        ([127, 0, 127], "Magenta"),
        ([255, 0, 255], "Magenta"),
        ([0, 0, 0], "Preto"),
        ([127, 127, 127], "Cinza"),
        ([255, 255, 255], "Branco"),
    ]

    # Extração automática de cores
    colors = sort(unique(last.(training_data)))
    color_idx = Dict(color => i for (i, color) in enumerate(colors))

    # Normalização e one-hot encoding
    normalized_data = [
        (Float64.(rgb)./255.0, zeros(length(colors)))
        for (rgb, color) in training_data
    ]

    for i in 1:length(training_data)
        color = training_data[i][2]
        normalized_data[i][2][color_idx[color]] = 1.0
    end

    # Parâmetros da rede
    mlp = MLP(3, 32, length(colors), 0.01)
    epochs = 10000

    # Treinamento
    for epoch in 1:epochs
        for (inputs, target) in normalized_data
            train!(mlp, inputs, target)
        end

        if epoch % 1000 == 0
            loss = 0.0
            for (inputs, target) in normalized_data
                _, output = forward(mlp, inputs)
                loss += sum((output .- target).^2)
            end
            @printf("Epoch %5d | Loss: %.4f\n", epoch, loss)
        end
    end

    # Teste final
    teste_data_input = [
        ([200, 0, 70]),
        ([165, 156, 159]),
    ]

    @printf("\n\n================ TESTE ===============\n")
    for rgb in teste_data_input
        inputs = Float64.(rgb) ./ 255.0
        _, output = forward(mlp, inputs)
        predicted = colors[argmax(output)]
        confidence = maximum(output)
        @printf("Cor RGB: %-15s | Previsão: %-10s | Confiança: %.1f%%\n", rgb, predicted, confidence*100)
    end
end

main()


## Execute ##
# $ cd .\perceptronxor\src\
# $ julia perceptronxor_4.jl
# $ mlp_reconhecimento_de_cor.jl

# Fiz algumas modificaçõe e funcionou perfeitamente meu DeepLearning, ficou parecendo um LSTM . Obrigado por sua ajuda. Abraços. \\