using LinearAlgebra
using Random
using Statistics
using Serialization
using Luxor
using Images
using Distributions
using FileIO

# Activation functions
relu(x) = max(0.0, x)
relu_derivative(x) = x > 0.0 ? 1.0 : 0.0
softmax(x) = exp.(x ./ 2.0 .- maximum(x ./ 2.0)) ./ sum(exp.(x ./ 2.0 .- maximum(x ./ 2.0)))  # Temperature scaling (T=2.0)

# Loss function (cross-entropy)
cross_entropy_loss(output, target) = -sum(target .* log.(output .+ 1e-8))

# CNN Structure
mutable struct CNN
    conv_filters::Int
    filter_size::Int
    output_size::Int
    conv_weights::Array{Float64, 3}
    conv_bias::Vector{Float64}
    fc_weights::Matrix{Float64}
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
    normal = Normal(0.0, sqrt(2.0 / (filter_size * filter_size)))
    conv_weights = rand(rng, normal, (conv_filters, filter_size, filter_size))
    conv_bias = zeros(Float64, conv_filters)
    fc_input_size = conv_filters * 12 * 12
    fc_normal = Normal(0.0, sqrt(2.0 / fc_input_size))
    fc_weights = rand(rng, fc_normal, (output_size, fc_input_size))
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

function forward(cnn::CNN, image::Matrix{Float64})
    conv_outputs = [convolve2d(image, cnn.conv_weights[k, :, :], cnn.conv_bias[k]) for k in 1:cnn.conv_filters]
    pooled_outputs = [maxpool2d(conv_out) for conv_out in conv_outputs]
    flattened = vcat([vec(pooled) for pooled in pooled_outputs]...)
    output = cnn.fc_weights * flattened + cnn.fc_bias
    output = softmax(output)
    return output, conv_outputs, pooled_outputs, flattened
end

function train!(cnn::CNN, image::Matrix{Float64}, target::Vector{Float64})
    output, conv_outputs, pooled_outputs, flattened = forward(cnn, image)
    output_error = output - target
    fc_weight_grad = reshape(output_error, :, 1) * reshape(flattened, 1, :)
    fc_bias_grad = output_error
    fc_error = cnn.fc_weights' * output_error
    pool_size = length(pooled_outputs[1])
    fc_error_reshaped = reshape(fc_error, cnn.conv_filters, pool_size)

    for k in 1:cnn.conv_filters
        pooled_error = reshape(fc_error_reshaped[k, :], size(pooled_outputs[k]))
        pool_h, pool_w = size(pooled_outputs[k])
        conv_output = conv_outputs[k]
        unpool_error = zeros(size(conv_output))
        for i in 1:pool_h
            for j in 1:pool_w
                region = conv_output[2i-1:2i, 2j-1:2j]
                max_idx = argmax(vec(region))
                unpool_error[2i-1+div(max_idx-1, 2), 2j-1+mod(max_idx-1, 2)] = pooled_error[i, j]
            end
        end
        conv_filter_grad = zeros(size(cnn.conv_weights[k, :, :]))
        filter_size = size(conv_filter_grad, 1)
        conv_h, conv_w = size(conv_output)
        for i in 1:conv_h - filter_size + 1
            for j in 1:conv_w - filter_size + 1
                region = image[i:i+filter_size-1, j:j+filter_size-1]
                conv_filter_grad .+= region .* unpool_error[i:i+filter_size-1, j:j+filter_size-1]
            end
        end
        cnn.conv_weight_vel[k, :, :] = cnn.momentum * cnn.conv_weight_vel[k, :, :] - cnn.learning_rate * conv_filter_grad
        cnn.conv_weights[k, :, :] += cnn.conv_weight_vel[k, :, :]
        cnn.conv_bias_vel[k] = cnn.momentum * cnn.conv_bias_vel[k] - cnn.learning_rate * sum(unpool_error)
        cnn.conv_bias[k] += cnn.conv_bias_vel[k]
    end

    cnn.fc_weight_vel = cnn.momentum * cnn.fc_weight_vel - cnn.learning_rate * fc_weight_grad - 0.0001 * cnn.fc_weights
    cnn.fc_weights += cnn.fc_weight_vel
    cnn.fc_bias_vel = cnn.momentum * cnn.fc_bias_vel - cnn.learning_rate * fc_bias_grad
    cnn.fc_bias += cnn.fc_bias_vel
end

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
    println("Raw output: ", round.(output, digits=4))
    idx = argmax(output)
    predicted = classifier.char_classes[idx]
    confidence = output[idx] * 100.0
    return predicted, confidence
end

function generate_char_image(ch::Char, font_path::String, size::Int=28)
    img = @imagematrix begin
        background("white")
        fontsize(size * 0.7)
        fontface(font_path)
        sethue("black")
        text(string(ch), Point(0, 0), halign=:center, valign=:middle)
    end size size
    return Float64.(Gray.(img))
end

function normalize_image(img::Array{Float64, 2})
    return 1.0 .- img
end

function list_fonts(font_dir::String)
    fonts = []
    for file in readdir(font_dir; join=true)
        if endswith(file, ".ttf")
            push!(fonts, file)
        end
    end
    return fonts
end

function prepare_data(char_classes::Vector{Char}, font_dir::String)
    fonts = list_fonts(font_dir)
    training_data = []
    rng = MersenneTwister(42)
    for ch in char_classes
        for font_path in fonts
            img = generate_char_image(ch, font_path)
            input = normalize_image(img) .+ randn(rng, 28, 28) * 0.1
            input = clamp.(input, 0.0, 1.0)
            target = zeros(length(char_classes))
            target[findfirst(==(ch), char_classes)] = 1.0
            push!(training_data, (input, target))
        end
    end
    return training_data
end

function train_model()
    model_path = "dados/char_classifier.jls"
    mkpath(dirname(model_path))

    println("Generating and training new model...")
    char_classes = collect("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.:,;'\"(!?)+-*/=")
    font_dir = "dados/FontsTrain"
    fonts = list_fonts(font_dir)
    println("Number of training fonts: ", length(fonts))
    training_data = prepare_data(char_classes, font_dir)
    cnn = CNN(16, 5, length(char_classes); learning_rate=0.001)
    epochs = 200

    for epoch in 1:epochs
        total_loss = 0.0
        correct = 0
        total = 0

        for (image, target) in training_data
            train!(cnn, image, target)
            output, _, _, _ = forward(cnn, image)
            total_loss += cross_entropy_loss(output, target)
            predicted_idx = argmax(output)
            true_idx = argmax(target)
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

function test_model()
    model_path = "dados/char_classifier.jls"
    classifier = load(model_path)
    println("Loaded model from $model_path")

    test_chars = ['A', 'B', 'C', 'J', 'K', 'L', 'X', 'Y', 'Z']
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

function reconhecercaracter(image_path::String)
    model_path = "dados/char_classifier.jls"
    if !isfile(model_path)
        println("Error: Trained model not found at $model_path. Run 'treino' first.")
        return
    end

    if !isfile(image_path)
        println("Error: Image file not found at $image_path.")
        return
    end

    classifier = load(model_path)
    println("Loaded model from $model_path")
    println("Attempting to load image from: ", abspath(image_path))

    try
        img = FileIO.load(image_path)
        println("Loaded image type: ", typeof(img))
        println("Loaded image size: ", size(img))
        img_gray = Gray.(img)
        img_resized = imresize(img_gray, (28, 28))
        img_matrix = Float64.(img_resized)
        input = normalize_image(img_matrix)
        Images.save("preprocessed_image.png", Gray.(input))  # Explicitly use Images.save

        predicted, confidence = predict(classifier, input)
        println("Predicted character: $predicted | Confidence: $(round(confidence, digits=1))%")
        println("Preprocessed image saved as 'preprocessed_image.png'")
    catch e
        println("Error processing image: ", e)
    end
end

function main(args)
    if length(args) > 0
        if args[1] == "treino"
            train_model()
        elseif args[1] == "reconhecer"
            test_model()
        elseif args[1] == "reconhecercaracter"
            if length(args) < 2
                println("Usage: julia cnn_reconhece_texto.jl reconhecercaracter <image_path>")
            else
                reconhecercaracter(args[2])
            end
        else
            println("Usage: julia cnn_reconhece_texto.jl [treino|reconhecer|reconhecercaracter <image_path>]")
        end
    else
        println("Usage: julia cnn_reconhece_texto.jl [treino|reconhecer|reconhecercaracter <image_path>]")
    end
end

main(ARGS)


# julia cnn_reconhece_texto.jl treino
# julia cnn_reconhece_texto.jl reconhecer
# julia cnn_reconhece_texto.jl reconhecercaracter
# julia cnn_reconhece_texto.jl reconhecercaracter images.png