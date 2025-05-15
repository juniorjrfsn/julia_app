using LinearAlgebra
using Random
using Statistics
using Serialization
using Luxor
using Images
using Distributions

# Activation functions
relu(x) = max(0.0, x)
relu_derivative(x) = x > 0.0 ? 1.0 : 0.0
softmax(x) = exp.(x .- maximum(x)) ./ sum(exp.(x .- maximum(x)))  # Numerically stable softmax

# Loss function (cross-entropy)
cross_entropy_loss(output, target) = -sum(target .* log.(output .+ 1e-8))

# CNN Structure
mutable struct CNN
    conv_filters::Int
    filter_size::Int
    output_size::Int
    conv_weights::Array{Float64, 3}  # Filters for convolution
    conv_bias::Vector{Float64}
    fc_weights::Matrix{Float64}      # Fully connected layer weights
    fc_bias::Vector{Float64}
    learning_rate::Float64
    momentum::Float64
    conv_weight_vel::Array{Float64, 3}
    conv_bias_vel::Vector{Float64}
    fc_weight_vel::Matrix{Float64}
    fc_bias_vel::Vector{Float64}
end

function CNN(conv_filters::Int, filter_size::Int, output_size::Int; learning_rate=0.001, momentum=0.9)
    rng = MersenneTwister(42)
    normal = Normal(0.0, sqrt(2.0 / (filter_size * filter_size)))  # Distribuição normal
    conv_weights = rand(rng, normal, (conv_filters, filter_size, filter_size))  # Pesos iniciais
    conv_bias = zeros(Float64, conv_filters)
    fc_input_size = conv_filters * 12 * 12  # Após pooling (28x28 -> 12x12)
    fc_normal = Normal(0.0, sqrt(2.0 / fc_input_size))  # Distribuição normal para camada fully connected
    fc_weights = rand(rng, fc_normal, (output_size, fc_input_size))  # Pesos iniciais
    fc_bias = zeros(Float64, output_size)

    CNN(
        conv_filters,
        filter_size,
        output_size,
        conv_weights,
        conv_bias,
        fc_weights,
        fc_bias,
        learning_rate,
        momentum,
        zeros(Float64, size(conv_weights)),
        zeros(Float64, conv_filters),
        zeros(Float64, size(fc_weights)),
        zeros(Float64, output_size)
    )
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
            output[i, j] = relu(sum(region .* filter) + bias)
        end
    end
    return output
end

# Max pooling (2x2)
function maxpool2d(input::Matrix{Float64})
    h, w = size(input)
    out_h, out_w = div(h, 2), div(w, 2)
    output = zeros(out_h, out_w)
    for i in 1:out_h
        for j in 1:out_w
            region = input[2i-1:2i, 2j-1:2j]
            output[i, j] = maximum(region)
        end
    end
    return output
end

# Forward pass
function forward(cnn::CNN, image::Matrix{Float64})
    # Convolution
    conv_outputs = [convolve2d(image, cnn.conv_weights[k, :, :], cnn.conv_bias[k]) for k in 1:cnn.conv_filters]
    # Pooling
    pooled_outputs = [maxpool2d(conv_out) for conv_out in conv_outputs]
    # Flatten
    flattened = vcat([vec(pooled) for pooled in pooled_outputs]...)
    # Fully connected layer
    output = cnn.fc_weights * flattened + cnn.fc_bias
    output = softmax(output)
    return output, conv_outputs, pooled_outputs, flattened
end

# Backward pass and parameter updates
function train!(cnn::CNN, image::Matrix{Float64}, target::Vector{Float64})
    # Forward pass
    output, conv_outputs, pooled_outputs, flattened = forward(cnn, image)

    # Compute output errors (cross-entropy loss gradient)
    output_error = output - target

    # Backpropagate through fully connected layer
    fc_weight_grad = reshape(output_error, :, 1) * reshape(flattened, 1, :)
    fc_bias_grad = output_error
    fc_error = cnn.fc_weights' * output_error

    # Reshape hidden errors to match pooled outputs
    pool_size = length(pooled_outputs[1])
    fc_error_reshaped = reshape(fc_error, cnn.conv_filters, pool_size)

    # Backpropagate through pooling and convolution layers
    for k in 1:cnn.conv_filters
        pooled_error = reshape(fc_error_reshaped[k, :], size(pooled_outputs[k]))
        pool_h, pool_w = size(pooled_outputs[k])
        conv_output = conv_outputs[k]

        # Unpooling
        unpool_error = zeros(size(conv_output))
        for i in 1:pool_h
            for j in 1:pool_w
                region = conv_output[2i-1:2i, 2j-1:2j]
                max_idx = argmax(vec(region))
                unpool_error[2i-1+div(max_idx-1, 2), 2j-1+mod(max_idx-1, 2)] = pooled_error[i, j]
            end
        end

        # Convolution gradient - Use conv_output size
        conv_filter_grad = zeros(size(cnn.conv_weights[k, :, :]))
        filter_size = size(conv_filter_grad, 1)  # Tamanho do filtro (5x5)
        conv_h, conv_w = size(conv_output)  # 24×24
        for i in 1:conv_h - filter_size + 1  # 1:20
            for j in 1:conv_w - filter_size + 1  # 1:20
                region = image[i:i+filter_size-1, j:j+filter_size-1]
                conv_filter_grad .+= region .* unpool_error[i:i+filter_size-1, j:j+filter_size-1]
            end
        end

        # Update convolutional parameters
        cnn.conv_weight_vel[k, :, :] = cnn.momentum * cnn.conv_weight_vel[k, :, :] - cnn.learning_rate * conv_filter_grad
        cnn.conv_weights[k, :, :] += cnn.conv_weight_vel[k, :, :]
        cnn.conv_bias_vel[k] = cnn.momentum * cnn.conv_bias_vel[k] - cnn.learning_rate * sum(unpool_error)
        cnn.conv_bias[k] += cnn.conv_bias_vel[k]
    end

    # Update fully connected parameters
    cnn.fc_weight_vel = cnn.momentum * cnn.fc_weight_vel - cnn.learning_rate * fc_weight_grad
    cnn.fc_weights += cnn.fc_weight_vel
    cnn.fc_bias_vel = cnn.momentum * cnn.fc_bias_vel - cnn.learning_rate * fc_bias_grad
    cnn.fc_bias += cnn.fc_bias_vel
end

# Character Classifier
mutable struct CharacterClassifier
    cnn::CNN
    char_classes::Vector{Char}
end

function save(classifier::CharacterClassifier, path::String)
    serialize(path, classifier)
end

function load(path::String)
    deserialize(path)
end

function predict(classifier::CharacterClassifier, image::Matrix{Float64})
    output, _, _, _ = forward(classifier.cnn, image)
    # println("Raw output: ", round.(output, digits=4))  # Debug softmax output
    idx = argmax(output)
    predicted = classifier.char_classes[idx]
    confidence = output[idx] * 100.0
    return predicted, confidence
end

# Generate character images using Luxor.jl
function generate_char_image(ch::Char, font_path::String, size::Int=28)
    img = @imagematrix begin
        background("white")  # Fundo branco
        fontsize(size * 0.7)  # Tamanho da fonte
        fontface(font_path)   # Caminho para a fonte
        sethue("black")       # Cor do texto
        text(string(ch), Point(0, 0), halign=:center, valign=:middle)  # Centralize o texto
    end size size

    # Converta a imagem para uma matriz de pixels
    return Float64.(Gray.(img))
end

# Normalize the image
function normalize_image(img::Array{Float64, 2})
    return 1.0 .- img  # Inverta as cores (preto -> branco e vice-versa)
end

# List fonts in a directory
function list_fonts(font_dir::String)
    fonts = []
    for file in readdir(font_dir; join=true)
        if endswith(file, ".ttf")
            push!(fonts, file)
        end
    end
    return fonts
end

# Prepare training data
function prepare_data(char_classes::Vector{Char}, font_dir::String)
    fonts = list_fonts(font_dir)
    training_data = []
    for ch in char_classes
        for font_path in fonts
            img = generate_char_image(ch, font_path)
            input = normalize_image(img)
            target = zeros(length(char_classes))
            target[findfirst(==(ch), char_classes)] = 1.0
            push!(training_data, (input, target))
        end
    end
    return training_data
end

# Train Model
function train_model()
    model_path = "dados/char_classifier.jls"
    mkpath(dirname(model_path))

    println("Generating and training new model...")
    char_classes = collect("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.:,;'\"(!?)+-*/=")
    font_dir = "dados/FontsTrain"
    fonts = list_fonts(font_dir)
    println("Number of training fonts: ", length(fonts))  # Check font variety
    training_data = prepare_data(char_classes, font_dir)
    cnn = CNN(16, 5, length(char_classes); learning_rate=0.001)
    epochs = 100

    for epoch in 1:epochs
        total_loss = 0.0
        correct = 0
        total = 0

        for (image, target) in training_data
            train!(cnn, image, target)
            output, _, _, _ = forward(cnn, image)
            total_loss += cross_entropy_loss(output, target)
            predicted_idx = argmax(output)  # Predicted class index
            true_idx = argmax(target)      # True class index
            if predicted_idx == true_idx
                correct += 1
            end
            total += 1
        end

        accuracy = correct / total
        println("Epoch $(lpad(epoch, 4)) | Loss: $(round(total_loss / total, digits=4)) | Accuracy: $(round(accuracy * 100, digits=2))%")
    end

    classifier = CharacterClassifier(cnn, char_classes)
    save(classifier, model_path)
    println("Model saved to $model_path")
end

# Test Model
function test_model()
    model_path = "dados/char_classifier.jls"
    classifier = load(model_path)
    println("Loaded model from $model_path")

    test_chars = ['A', 'B', 'C', 'F','J', 'K', 'L', 'X', 'Y', 'Z']
    font_dir = "dados/FontsTest"
    fonts = list_fonts(font_dir)

    correct = 0
    total = 0

    for ch in test_chars
        for font_path in fonts
            img = generate_char_image(ch, font_path)
            input = normalize_image(img)
            predicted, confidence = predict(classifier, input)
            println("Char: $ch (Font: $font_path) | Predicted: $(predicted) | Confidence: $(round(confidence, digits=1))%")
            if ch == predicted
                correct += 1
            end
            total += 1
        end
    end

    accuracy = (correct / total) * 100
    println("Test Accuracy: $(round(accuracy, digits=2))% ($correct out of $total)")
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



# julia cnn_reconhece_texto_otimo.jl treino
# julia cnn_reconhece_texto_otimo.jl reconhecer
