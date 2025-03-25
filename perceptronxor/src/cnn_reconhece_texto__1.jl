using LinearAlgebra
using Random
using Statistics
using Serialization
using Images
using FileIO

# Activation functions
relu(x) = max(0.0, x)
relu_derivative(x) = x > 0.0 ? 1.0 : 0.0
softmax(x) = exp.(x .- maximum(x)) ./ sum(exp.(x .- maximum(x)))  # Numerically stable softmax

# Loss function (cross-entropy)
cross_entropy_loss(output, target) = -sum(target .* log.(output .+ 1e-8))

# CNN Structure
mutable struct CNN
    input_channels::Int
    conv_filters::Int
    filter_size::Int
    hidden_size::Int
    output_size::Int
    conv_weights::Array{Float64, 3}  # Filters for convolution
    conv_bias::Vector{Float64}
    fc_weights::Matrix{Float64}      # Fully connected layer weights
    fc_bias::Vector{Float64}
    learning_rate::Float64
end

function CNN(input_channels::Int, conv_filters::Int, filter_size::Int, hidden_size::Int, output_size::Int, learning_rate::Float64)
    rng = MersenneTwister(42)
    conv_weights = randn(rng, filter_size, filter_size, conv_filters) * sqrt(1 / (filter_size * filter_size))
    conv_bias = zeros(conv_filters)
    fc_input_size = conv_filters * (14 * 14)  # After pooling (28x28 -> 14x14)
    fc_weights = randn(rng, hidden_size, fc_input_size) * sqrt(1 / fc_input_size)
    fc_bias = zeros(hidden_size)
    return CNN(input_channels, conv_filters, filter_size, hidden_size, output_size, conv_weights, conv_bias, fc_weights, fc_bias, learning_rate)
end

# Convolution operation
function convolve2d(image::Matrix{Float64}, filter::Matrix{Float64}, bias::Float64)
    h, w = size(image)
    fh, fw = size(filter)
    out_h, out_w = h - fh + 1, w - fw + 1
    output = zeros(out_h, out_w)
    for i in 1:out_h
        for j in 1:out_w
            region = image[i:i+fh-1, j:j+fw-1]
            output[i, j] = sum(region .* filter) + bias
        end
    end
    return relu.(output)
end

# Max pooling (2x2)
function maxpool2d(input::Matrix{Float64})
    h, w = size(input)
    out_h, out_w = h ÷ 2, w ÷ 2
    output = zeros(out_h, out_w)
    for i in 1:out_h
        for j in 1:out_w
            region = input[2*i-1:2*i, 2*j-1:2*j]
            output[i, j] = maximum(region)
        end
    end
    return output
end

# Forward pass
function forward(cnn::CNN, image::Matrix{Float64})
    # Convolution
    conv_outputs = [convolve2d(image, cnn.conv_weights[:, :, k], cnn.conv_bias[k]) for k in 1:cnn.conv_filters]
    # Pooling
    pooled_outputs = [maxpool2d(conv_out) for conv_out in conv_outputs]
    # Flatten
    flattened = vcat([vec(pooled) for pooled in pooled_outputs]...)
    # Fully connected layer
    hidden_inputs = cnn.fc_weights * flattened .+ cnn.fc_bias
    hidden_outputs = relu.(hidden_inputs)
    # Output layer
    output = softmax(hidden_outputs)
    return conv_outputs, pooled_outputs, flattened, hidden_outputs, output
end

# Backward pass and parameter updates
function train(cnn::CNN, image::Matrix{Float64}, target::Vector{Float64})
    # Forward pass
    conv_outputs, pooled_outputs, flattened, hidden_outputs, output = forward(cnn, image)

    # Compute output errors (cross-entropy loss gradient)
    output_errors = output .- target

    # Backpropagate through fully connected layer
    fc_weight_grad = output_errors * hidden_outputs'
    fc_bias_grad = output_errors
    hidden_errors = cnn.fc_weights' * output_errors .* relu_derivative.(hidden_outputs)

    # Reshape hidden errors to match pooled outputs
    pooled_errors = reshape(hidden_errors, (14, 14, cnn.conv_filters))

    # Backpropagate through pooling and convolution layers
    for k in 1:cnn.conv_filters
        pooled_error = pooled_errors[:, :, k]
        pool_h, pool_w = size(pooled_outputs[k])
        conv_output = conv_outputs[k]

        # Unpooling
        unpool_error = zeros(size(conv_output))
        for i in 1:pool_h
            for j in 1:pool_w
                region = conv_output[2*i-1:2*i, 2*j-1:2*j]
                max_idx = argmax(region)
                unpool_error[2*i-1:2*i, 2*j-1:2*j][max_idx] = pooled_error[i, j]
            end
        end

        # Convolution gradient
        conv_filter_grad = zeros(size(cnn.conv_weights[:, :, k]))
        for i in 1:size(conv_filter_grad, 1)
            for j in 1:size(conv_filter_grad, 2)
                region = image[i:i+size(conv_filter_grad, 1)-1, j:j+size(conv_filter_grad, 2)-1]
                conv_filter_grad[i, j] = sum(region .* unpool_error)
            end
        end

        # Update convolutional parameters
        cnn.conv_weights[:, :, k] .-= cnn.learning_rate * conv_filter_grad
        cnn.conv_bias[k] -= cnn.learning_rate * sum(unpool_error)
    end

    # Update fully connected parameters
    cnn.fc_weights .-= cnn.learning_rate * fc_weight_grad
    cnn.fc_bias .-= cnn.learning_rate * fc_bias_grad
end

# Character Classifier
mutable struct CharacterClassifier
    cnn::CNN
    char_classes::Vector{String}
end

function save(classifier::CharacterClassifier, path::String)
    serialize(path, classifier)
end

function load(path::String)
    deserialize(path)
end

function predict(classifier::CharacterClassifier, image::Matrix{Float64})
    _, _, _, _, output = forward(classifier.cnn, image)
    max_idx = argmax(output)
    predicted = classifier.char_classes[max_idx]
    confidence = output[max_idx] * 100.0
    return predicted, confidence
end

# Generate character images
function generate_char_image(char::Char, font::String, size::Int=28)
    img = rand(Float64, size, size) * 0.1  # Noise background
    img[10:18, 10:18] .= 1.0  # Simplified "character" square
    return img
end

function generate_training_data()
    chars = ['A':'Z'; 'a':'z'; '0':'9'; ['!', '@', '#', '$', '%', '&', '*', '(', ')', '-', '+', '=', '[', ']', '{', '}', '|', ';', ':', ',', '.', '/', '?']]
    fonts = ["Verdana", "Serif", "Times New Roman", "Courier New"]
    data_dir = "dados/imagens"
    mkpath(data_dir)

    training_data = []
    for char in chars
        for font in fonts
            img = generate_char_image(char, font)
            path = joinpath(data_dir, "$(char)_$(font).png")

            # Convert the image to a Gray image and save it
            gray_img = reinterpret(Gray{Float64}, img)  # Convert to a compatible format
            save(path, gray_img)  # Save the image

            push!(training_data, (img, string(char)))
        end
    end
    return training_data
end

# Training Data Preparation
function prepare_data()
    training_data = generate_training_data()
    chars = sort(unique(last.(training_data)))
    char_idx = Dict(char => idx for (idx, char) in enumerate(chars))
    normalized_data = [(img, zeros(length(chars)) |> (t -> (t[char_idx[char]] = 1.0; t))) 
                      for (img, char) in training_data]
    return normalized_data, chars
end

# Train Model
function train_model()
    model_path = "dados/char_classifier.jls"
    mkpath(dirname(model_path))
    
    println("Generating and training new model...")
    normalized_data, chars = prepare_data()
    cnn = CNN(1, 8, 5, 128, length(chars), 0.001)  # 1 input channel (grayscale), 8 filters, 5x5 filter
    epochs = 1000
    
    for epoch in 0:epochs-1
        total_loss = 0.0
        for (img, target) in normalized_data
            train(cnn, img, target)
            _, _, _, _, output = forward(cnn, img)
            total_loss += cross_entropy_loss(output, target)
        end
        if epoch % 100 == 0
            avg_loss = total_loss / length(normalized_data)
            println("Epoch $(lpad(epoch, 4)) | Average Loss: $(round(avg_loss, digits=4))")
        end
    end
    
    classifier = CharacterClassifier(cnn, chars)
    save(classifier, model_path)
    println("Model saved to $model_path")
end

# Test Model
function test_model()
    model_path = "dados/char_classifier.jls"
    classifier = load(model_path)
    println("Loaded model from $model_path")

    test_chars = ['A', '1', '#', 'z']
    test_fonts = ["Verdana", "Courier New"]
    println("\n================ TESTE ===============")
    for char in test_chars
        for font in test_fonts
            img = generate_char_image(char, font)
            predicted, confidence = predict(classifier, img)
            println("Char: $char (Font: $font) | Previsão: $(lpad(predicted, 2)) | Confiança: $(round(confidence, digits=1))%")
        end
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
            println("Usage: julia cnn_reconhece_texto.jl [treino|reconhecer]")
        end
    else
        println("Usage: julia cnn_reconhece_texto.jl [treino|reconhecer]")
    end
end

main(ARGS)



# julia cnn_reconhece_texto__1.jl treino
# julia cnn_reconhece_texto__1.jl reconhecer
