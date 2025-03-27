using LinearAlgebra
using Random
using Statistics
using Serialization
using Luxor
using Images
using ImageMagick
using Distributions

# === Activation and Loss Functions ===
relu(x) = max(0.0, x)
relu_derivative(x) = x > 0.0 ? 1.0 : 0.0
softmax(x) = exp.(x .- maximum(x)) ./ sum(exp.(x .- maximum(x)))
cross_entropy_loss(output, target) = -sum(target .* log.(output .+ 1e-8))

# === CNN Structure ===
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
    fc_input_size = conv_filters * 12 * 12  # After pooling (28x28 -> 12x12)
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

# === Convolution and Pooling Operations ===
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

# === Forward Pass ===
function forward(cnn::CNN, image::Matrix{Float64})
    conv_outputs = [convolve2d(image, cnn.conv_weights[k, :, :], cnn.conv_bias[k]) for k in 1:cnn.conv_filters]
    pooled_outputs = [maxpool2d(conv_out) for conv_out in conv_outputs]
    flattened = vcat([vec(pooled) for pooled in pooled_outputs]...)
    output = cnn.fc_weights * flattened + cnn.fc_bias
    output = softmax(output)
    return output, conv_outputs, pooled_outputs, flattened
end

# === Backward Pass ===
function train!(cnn::CNN, image::Matrix{Float64}, target::Vector{Float64})
    output, conv_outputs, pooled_outputs, flattened = forward(cnn, image)
    output_error = output - target

    # Fully connected layer gradients
    fc_weight_grad = output_error * flattened'
    fc_bias_grad = output_error

    # Backpropagate error
    fc_error = cnn.fc_weights' * output_error
    pool_size = length(pooled_outputs[1])
    fc_error_reshaped = reshape(fc_error, cnn.conv_filters, pool_size)

    # Convolutional layer gradients
    for k in 1:cnn.conv_filters
        pooled_error = reshape(fc_error_reshaped[k, :], size(pooled_outputs[k]))
        conv_output = conv_outputs[k]
        unpool_error = zeros(size(conv_output))

        # Unpooling
        pool_h, pool_w = size(pooled_outputs[k])
        for i in 1:pool_h
            for j in 1:pool_w
                region = conv_output[2i-1:2i, 2j-1:2j]
                max_idx = argmax(vec(region))
                unpool_error[2i-1+div(max_idx-1, 2), 2j-1+mod(max_idx-1, 2)] = pooled_error[i, j]
            end
        end

        # Compute gradients for conv filters
        filter_size = cnn.filter_size
        conv_filter_grad = zeros(size(cnn.conv_weights[k, :, :]))
        conv_h, conv_w = size(conv_output)
        for i in 1:conv_h-filter_size+1
            for j in 1:conv_w-filter_size+1
                region = image[i:i+filter_size-1, j:j+filter_size-1]
                conv_filter_grad .+= region .* unpool_error[i, j]
            end
        end

        # Update parameters with momentum
        cnn.conv_weight_vel[k, :, :] = cnn.momentum * cnn.conv_weight_vel[k, :, :] - cnn.learning_rate * conv_filter_grad
        cnn.conv_weights[k, :, :] += cnn.conv_weight_vel[k, :, :]
        cnn.conv_bias_vel[k] = cnn.momentum * cnn.conv_bias_vel[k] - cnn.learning_rate * sum(unpool_error)
        cnn.conv_bias[k] += cnn.conv_bias_vel[k]
    end

    # Update fully connected layer
    cnn.fc_weight_vel = cnn.momentum * cnn.fc_weight_vel - cnn.learning_rate * fc_weight_grad
    cnn.fc_weights += cnn.fc_weight_vel
    cnn.fc_bias_vel = cnn.momentum * cnn.fc_bias_vel - cnn.learning_rate * fc_bias_grad
    cnn.fc_bias += cnn.fc_bias_vel
end

# === Character Classifier ===
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
    idx = argmax(output)
    return classifier.char_classes[idx], output[idx] * 100.0
end

# === Image Generation ===
function generate_char_image(ch::Char, font_path::String, size::Int=28)
    if !isfile(font_path)
        error("Font file not found: $font_path")
    end
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
    return filter(f -> endswith(f, ".ttf"), readdir(font_dir; join=true))
end

function generate_images_from_fonts(char_classes::Vector{Char}, font_dir::String, output_dir::String, size::Int=28)
    mkpath(output_dir)
    fonts = list_fonts(font_dir)
    for ch in char_classes
        for font_path in fonts
            try
                img = generate_char_image(ch, font_path, size)
                img_normalized = normalize_image(img)
                img_to_save = Gray{N0f8}.(clamp.(img_normalized, 0, 1))
                font_name = splitext(basename(font_path))[1]
                save_path = joinpath(output_dir, "$(ch)_$(font_name).png")
                ImageMagick.save(save_path, img_to_save)
            catch e
                println("Error generating image for char: $ch, font: $font_path - $(e)")
            end
        end
    end
    println("Images generated and saved in $output_dir")
end

# === Data Preparation ===
function prepare_data_from_images(char_classes::Vector{Char}, image_dir::String)
    training_data = Tuple{Matrix{Float64}, Vector{Float64}}[]
    for file in readdir(image_dir; join=true)
        if endswith(file, ".png")
            try
                img = Float64.(Gray.(ImageMagick.load(file)))
                if size(img) != (28, 28)
                    continue  # Skip images with wrong dimensions
                end
                char = basename(file)[1]
                target = zeros(Float64, length(char_classes))
                idx = findfirst(==(char), char_classes)
                if idx !== nothing
                    target[idx] = 1.0
                    push!(training_data, (img, target))
                end
            catch e
                println("Error loading image $file: $e")
            end
        end
    end
    return shuffle(training_data)
end

# === Training ===
function train_model()
    model_path = "dados/char_classifier.jls"
    mkpath(dirname(model_path))
    println("Training new model...")

    char_classes = collect("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.:,;'\"(!?)+-*/=")
    image_train_dir = "dados/FontsImgTrain"

    # Generate images if directory is empty
    if !isdir(image_train_dir) || isempty(readdir(image_train_dir))
        generate_images_from_fonts(char_classes, "dados/FontsTrain", image_train_dir)
    end

    training_data = prepare_data_from_images(char_classes, image_train_dir)
    if isempty(training_data)
        error("No training data available. Please check image generation.")
    end

    cnn = CNN(16, 5, length(char_classes); learning_rate=0.001)
    epochs = 100

    for epoch in 1:epochs
        total_loss = 0.0
        correct = 0
        total = 0

        # Shuffle data each epoch
        shuffle!(training_data)

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
        avg_loss = total_loss / total
        println("Epoch $(lpad(epoch, 4)) | Loss: $(round(avg_loss, digits=4)) | Accuracy: $(round(accuracy * 100, digits=2))%")
    end

    classifier = CharacterClassifier(cnn, char_classes)
    save(classifier, model_path)
    println("Model saved to $model_path")
end

# === Testing ===
function test_model()
    model_path = "dados/char_classifier.jls"
    if !isfile(model_path)
        error("Model file not found. Please train the model first.")
    end

    classifier = load(model_path)
    println("Loaded model from $model_path")

    test_image_dir = "dados/FontsImgTest"
    if !isdir(test_image_dir)
        error("Test image directory not found.")
    end

    correct = 0
    total = 0
    for file in readdir(test_image_dir; join=true)
        if endswith(file, ".png")
            try
                img = Float64.(Gray.(ImageMagick.load(file)))
                if size(img) != (28, 28)
                    continue
                end
                ch = basename(file)[1]
                predicted, confidence = predict(classifier, img)
                println("Char: $ch | Predicted: $predicted | Confidence: $(round(confidence, digits=1))%")
                if ch == predicted
                    correct += 1
                end
                total += 1
            catch e
                println("Error testing image $file: $e")
            end
        end
    end
    accuracy = (correct / total) * 100
    println("Test Accuracy: $(round(accuracy, digits=2))% ($correct out of $total)")
end

# === Main Function ===
function main(args)
    if length(args) > 0
        if args[1] == "gerarimagens"
            char_classes = collect("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.:,;'\"(!?)+-*/=")
            generate_images_from_fonts(char_classes, "dados/FontsTrain", "dados/FontsImgTrain")
        elseif args[1] == "treino"
            train_model()
        elseif args[1] == "reconhecer"
            test_model()
        else
            println("Usage: julia cnn_reconhece_texto.jl [gerarimagens|treino|reconhecer]")
        end
    else
        println("Usage: julia cnn_reconhece_texto.jl [gerarimagens|treino|reconhecer]")
    end
end

main(ARGS)

# === Usage ===
# To generate images, train the model, or test the model, run the following commands in the terminal:
# julia cnn_reconhece_texto.jl gerarimagens
# julia cnn_reconhece_texto.jl treino
# julia cnn_reconhece_texto.jl reconhecer
# Make sure to have the necessary directories and font files in place before running the commands.
# The model will be saved in "dados/char_classifier.jls" and can be loaded for testing.

# The generated images will be saved in "dados/FontsImgTrain" and can be used for training.
# The test images should be placed in "dados/FontsImgTest" for testing the model.

# Ensure you have the required packages installed: Luxor, Images, ImageMagick, Distributions, Statistics
# You can install them using Julia's package manager:

# Finalmente uma IA que reconhece texto em imagens de forma simples e eficiente.
# A implementação é feita em Julia, uma linguagem de programação de alto desempenho e fácil de usar.