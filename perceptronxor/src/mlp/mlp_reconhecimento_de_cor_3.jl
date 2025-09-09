using Random
using Printf
using JLD2

# Definição da rede neural com múltiplas saídas
mutable struct MLP
    input_size::Int
    hidden_size::Int
    output_size::Int
    weights_input_hidden::Matrix{Float64}
    weights_hidden_output::Matrix{Float64}
    bias_hidden::Vector{Float64}
    bias_output::Vector{Float64}  # Agora é um vetor para múltiplas classes
    learning_rate::Float64
end

# Função de inicialização da rede
function MLP(input_size::Int, hidden_size::Int, output_size::Int, learning_rate::Float64)
    # Inicialização He-et-al para melhor convergência
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
    # Camada oculta
    hidden_inputs = mlp.weights_input_hidden' * inputs .+ mlp.bias_hidden
    hidden_outputs = sigmoid.(hidden_inputs)
    
    # Camada de saída
    output_inputs = mlp.weights_hidden_output' * hidden_outputs .+ mlp.bias_output
    output = softmax(output_inputs)
    
    return hidden_outputs, output
end

# Função de treinamento
function train!(mlp::MLP, inputs::Vector{Float64}, target::Vector{Float64})
    # Forward pass
    hidden_outputs, output = forward(mlp, inputs)
    
    # Backpropagation
    output_errors = output .- target  # Derivada da cross-entropy com softmax
    
    # Erros da camada oculta
    hidden_errors = mlp.weights_hidden_output * output_errors .* sigmoid_derivative(hidden_outputs)
    
    # Atualização dos pesos e biases
    # Camada oculta -> saída
    mlp.weights_hidden_output .-= mlp.learning_rate .* (hidden_outputs * output_errors')
    mlp.bias_output .-= mlp.learning_rate .* output_errors
    
    # Camada entrada -> oculta
    mlp.weights_input_hidden .-= mlp.learning_rate .* (inputs * hidden_errors')
    mlp.bias_hidden .-= mlp.learning_rate .* hidden_errors
end

# Mapeamento de cores para índices
const COLORS = [
    "Rose", "Vermelho", "Laranja", "Amarelo", "Primavera", "Verde",
    "Turquesa", "Ciano", "Cobalto", "Azul", "Violeta", "Magenta", "Preto", "Cinza", "Branco"
]
const COLOR_IDX = Dict(color => i for (i, color) in enumerate(COLORS))

# Função para converter nome da cor para one-hot
function one_hot(color::String)
    vec = zeros(length(COLORS))
    vec[COLOR_IDX[color]] = 1.0
    return vec
end

# Função para salvar o modelo
function save_model(mlp::MLP, filename::String)
    save(filename, "mlp", mlp)
end

# Função para carregar o modelo
function load_model(filename::String)
    return load(filename, "mlp")
end

function main(mode::String)
    Random.seed!(42)

    # Dados de treinamento normalizados (0-1)
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

    # Normalização dos dados de entrada
    normalized_data = [
        (Float64.(rgb)./255.0, one_hot(color))
        for (rgb, color) in training_data
    ]

    if mode == "treino"
        # Parâmetros da rede
        mlp = MLP(3, 32, length(COLORS), 0.01)
        epochs = 10000

        # Treinamento
        for epoch in 1:epochs
            for (inputs, target) in normalized_data
                train!(mlp, inputs, target)
            end

            # Exibe erro a cada 1000 épocas
            if epoch % 1000 == 0
                loss = 0.0
                for (inputs, target) in normalized_data
                    _, output = forward(mlp, inputs)
                    loss += sum((output .- target).^2)
                end
                @printf("Epoch %5d | Loss: %.4f\n", epoch, loss)
            end
        end

        # Salvar o modelo treinado
        save_model(mlp, "mlp_model.jld2")
        @printf("\nModelo treinado e salvo com sucesso!\n")

    elseif mode == "reconhecer"
        # Carregar o modelo treinado
        try
            mlp = load_model("mlp_model.jld2")
        catch e
            error("Erro ao carregar o modelo. Certifique-se de que o arquivo 'mlp_model.jld2' existe. Execute o modo 'treino' primeiro.")
        end

        # Dados de teste
        teste_data_input = [
            ([200, 0, 70]),
            ([165, 156, 255]),
        ]

        # Teste final
        @printf("\n\n================ TESTE ===============\n")
        for rgb in teste_data_input
            inputs = Float64.(rgb) ./ 255.0
            _, output = forward(mlp, inputs)
            predicted = COLORS[argmax(output)]
            confidence = maximum(output)
            @printf("Cor RGB: %-15s | Previsão: %-10s | Confiança: %.1f%%\n", rgb, predicted, confidence*100)
        end
    else
        error("Modo inválido. Use 'treino' ou 'reconhecer'.")
    end
end

# Verifica argumentos da linha de comando
if length(ARGS) != 1
    error("Forneça exatamente um argumento: 'treino' ou 'reconhecer'")
end

main(ARGS[1])


## Execute ##
# $ cd .\perceptronxor\src\
# $ julia perceptronxor_4.jl
# $ mlp_reconhecimento_de_cor_3.jl

# Fiz algumas modificaçõe e funcionou perfeitamente meu DeepLearning, ficou parecendo um LSTM . Obrigado por sua ajuda. Abraços. \\

# $  julia mlp_reconhecimento_de_cor_3.jl treino
# $  julia mlp_reconhecimento_de_cor_3.jl reconhecer
# MLP para reconhecimento de cores com múltiplas saídas