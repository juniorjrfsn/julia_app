using Random
using Printf

mutable struct MLP
    input_size::Int
    hidden_size::Int
    output_size::Int
    weights_input_hidden::Matrix{Float64}
    weights_hidden_output::Matrix{Float64}
    bias_hidden::Vector{Float64}
    bias_output::Float64
    learning_rate::Float64
end

function MLP(input_size::Int, hidden_size::Int, output_size::Int, learning_rate::Float64)
    # Initialize weights and biases with continuous values between -0.5 and 0.5
    weights_input_hidden = rand(input_size, hidden_size) .- 0.5
    weights_hidden_output = rand(hidden_size, output_size) .- 0.5
    bias_hidden = rand(hidden_size) .- 0.5
    bias_output = rand() - 0.5

    MLP(input_size, hidden_size, output_size, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, learning_rate)
end

sigmoid(x) = 1.0 / (1.0 + exp(-x))
sigmoid_derivative(x) = x * (1.0 - x)

function forward(mlp::MLP, inputs::Vector{Float64})
    hidden_outputs = [sigmoid(sum(inputs .* mlp.weights_input_hidden[:, j]) + mlp.bias_hidden[j]) for j in 1:mlp.hidden_size]
    output = sigmoid(sum(hidden_outputs .* vec(mlp.weights_hidden_output)) + mlp.bias_output)
    return output
end

function train!(mlp::MLP, inputs::Vector{Float64}, target::Float64)
    # Forward pass
    hidden_outputs = [sigmoid(sum(inputs .* mlp.weights_input_hidden[:, j]) + mlp.bias_hidden[j]) for j in 1:mlp.hidden_size]
    output = sigmoid(sum(hidden_outputs .* vec(mlp.weights_hidden_output)) + mlp.bias_output)

    # Backpropagation
    output_error = (target - output) * sigmoid_derivative(output)
    hidden_errors = [output_error * mlp.weights_hidden_output[j] * sigmoid_derivative(hidden_outputs[j]) for j in 1:mlp.hidden_size]

    # Update weights and biases
    for j in 1:mlp.hidden_size
        for i in 1:mlp.input_size
            mlp.weights_input_hidden[i, j] += mlp.learning_rate * hidden_errors[j] * inputs[i]
        end
        mlp.bias_hidden[j] += mlp.learning_rate * hidden_errors[j]
    end

    for j in 1:mlp.hidden_size
        mlp.weights_hidden_output[j] += mlp.learning_rate * output_error * hidden_outputs[j]
    end
    mlp.bias_output += mlp.learning_rate * output_error

end

function main()
    Random.seed!(123)
    mlp = MLP(2, 10, 1, 0.1)
    training_data = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ]

    for _ in 1:10000
        for (inputs, target) in training_data
            train!(mlp, inputs, target)
        end
    end

    # Teste
    @printf("\n\n================ TESTE ===============\n")
    threshold = 0.5
    resposta = ""

    for (inputs, target) in training_data
        output = forward(mlp, inputs )
        classification = output >= threshold ? 1 : 0
        linha = @sprintf("Entradas: %-10s | Alvo: %-2d | Saída: %-6.4f | XOR: Classificação: %d",
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
# $ julia perceptronxor_2.jl
