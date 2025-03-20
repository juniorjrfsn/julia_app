using LinearAlgebra
using Random
using Statistics
using Serialization

# Activation functions
relu(x) = max(0.0, x)
relu_derivative(x) = x > 0.0 ? 1.0 : 0.0

softmax(x) = exp.(x) ./ sum(exp.(x))

# MLP Structure
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

function MLP(input_size::Int, hidden_size::Int, output_size::Int, learning_rate::Float64)
    rng = MersenneTwister(42)
    weights_input_hidden = randn(rng, input_size, hidden_size) * sqrt(1 / input_size)
    weights_hidden_output = randn(rng, hidden_size, output_size) * sqrt(1 / hidden_size)
    bias_hidden = zeros(hidden_size)
    bias_output = zeros(output_size)
    return MLP(input_size, hidden_size, output_size, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, learning_rate)
end

function forward(mlp::MLP, inputs::Vector{Float64})
    hidden_inputs = mlp.weights_input_hidden' * inputs .+ mlp.bias_hidden
    hidden_outputs = relu.(hidden_inputs)
    output_inputs = mlp.weights_hidden_output' * hidden_outputs .+ mlp.bias_output
    output = softmax(output_inputs)
    return hidden_outputs, output
end

function train(mlp::MLP, inputs::Vector{Float64}, target::Vector{Float64})
    hidden_outputs, output = forward(mlp, inputs)
    output_errors = output .- target

    relu_deriv = relu_derivative.(hidden_outputs)
    hidden_errors = mlp.weights_hidden_output * output_errors .* relu_deriv

    # Update weights and biases
    mlp.weights_hidden_output .-= mlp.learning_rate * (hidden_outputs * output_errors')
    mlp.bias_output .-= mlp.learning_rate * output_errors

    mlp.weights_input_hidden .-= mlp.learning_rate * (inputs * hidden_errors')
    mlp.bias_hidden .-= mlp.learning_rate * hidden_errors
end

# Color Classifier
mutable struct ColorClassifier
    mlp::MLP
    color_classes::Vector{String}
end

function save(classifier::ColorClassifier, path::String)
    serialize(path, classifier)
end

function load(path::String)
    deserialize(path)
end

function predict(classifier::ColorClassifier, rgb::AbstractVector{<:Number})
    inputs = Float64.(rgb) ./ 255.0
    _, output = forward(classifier.mlp, inputs)
    max_idx = argmax(output)
    predicted = classifier.color_classes[max_idx]
    confidence = output[max_idx] * 100.0
    return predicted, confidence
end

# Training Data
training_data = [
    ([255, 0, 0], "Vermelho"),
    ([200, 0, 70], "Vermelho"),  # Additional example for Vermelho
    ([220, 0, 50], "Vermelho"),  # Another example for Vermelho
    ([255, 0, 127], "Rose"),
    ([200, 0, 100], "Rose"),     # Additional example for Rose
    ([30, 10, 240], "Azul"),
    ([0, 0, 255], "Azul"),
    ([127, 0, 0], "Vermelho"),
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

# Data Augmentation
function augment_data(rgb::Vector{Int}, label::String, n::Int)
    augmented = []
    for _ in 1:n
        noise = rand(-10:10, 3)  # Add small random noise
        new_rgb = clamp.(rgb .+ noise, 0, 255)  # Clamp values to [0, 255]
        push!(augmented, (new_rgb, label))
    end
    return augmented
end

augmented_data = vcat(
    training_data,
    augment_data([200, 0, 70], "Vermelho", 5)...,
    augment_data([255, 0, 127], "Rose", 5)...,
    augment_data([0, 0, 255], "Azul", 5)...
)
training_data = unique(augmented_data)

colors = sort(unique(last.(training_data)))
color_idx = Dict(color => idx for (idx, color) in enumerate(colors))

normalized_data = [
    (Float64.(rgb) ./ 255.0, zeros(length(colors)) |> (t -> (t[color_idx[color]] = 1.0; t)))
    for (rgb, color) in training_data
]

# Train Model
function train_model()
    model_path = "dados/color_classifier.jls"
    mkpath(dirname(model_path))

    println("Training new model...")
    mlp = MLP(3, 128, length(colors), 0.0005)  # Increased hidden layer size and reduced learning rate
    epochs = 20000  # Increased epochs

    for epoch in 0:epochs-1
        for (inputs, target) in normalized_data
            train(mlp, inputs, target)
        end

        if epoch % 1000 == 0
            loss = sum(-log(output[argmax(target)]) for (inputs, target) in normalized_data for (_, output) in [forward(mlp, inputs)])
            avg_confidence = mean(maximum(output) for (inputs, _) in normalized_data for (_, output) in [forward(mlp, inputs)])
            println("Epoch $(lpad(epoch, 5)) | Loss: $(round(loss, digits=4)) | Avg Confidence: $(round(avg_confidence * 100, digits=1))%")
        end
    end

    classifier = ColorClassifier(mlp, colors)
    save(classifier, model_path)
    println("Model saved to $model_path")
end

# Test Model
function test_model()
    model_path = "dados/color_classifier.jls"

    classifier = load(model_path)
    println("Loaded model from $model_path")

    test_data = [
        [200, 0, 70],
        [130, 10, 80],
    ]

    println("\n================ TESTE ===============")
    for rgb in test_data
        predicted, confidence = predict(classifier, rgb)
        println("Cor RGB: $rgb | Previsão: $(lpad(predicted, 10)) | Confiança: $(round(confidence, digits=1))%")
    end
end

# Main Function
function main(args)
    if length(args) > 0
        if args[1] == "treino"
            train_model()
        elseif args[1] == "reconhecer"
            test_model()
        else
            println("Usage: julia mlp_reconhecimento_de_cor_2.jl [treino|reconhecer]")
        end
    else
        println("Usage: julia mlp_reconhecimento_de_cor_2.jl [treino|reconhecer]")
    end
end

main(ARGS)

## Execute ##
# $ cd .\perceptronxor\src\

# $  julia mlp_reconhecimento_de_cor_2.jl treino
# $  julia mlp_reconhecimento_de_cor_2.jl reconhecer
