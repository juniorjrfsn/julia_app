# projeto: webcamcnn
# file: webcamcnn/src/core.jl



module CNNCheckinCore

using Flux
using Images
using FileIO
using CUDA
using Statistics
using Random
using JLD2
using TOML
using ImageTransformations
using LinearAlgebra
using Dates

# Global configurations
const IMG_SIZE = (128, 128)
const BATCH_SIZE = 8
const PRETRAIN_EPOCHS = 30
const INCREMENTAL_EPOCHS = 15
const LEARNING_RATE = 0.0001
const INCREMENTAL_LR = 0.00005

# Directory structure
const TRAIN_DATA_PATH = "../../../dados/webcamcnn/fotos_rosto"
const INCREMENTAL_DATA_PATH = "../../../dados/webcamcnn/fotos_new" 
const AUTH_DATA_PATH = "../../../dados/webcamcnn/fotos_auth"

# Model files
const MODEL_PATH = "../../../dados/webcamcnn/face_recognition_model.jld2"
const CONFIG_PATH = "../../../dados/webcamcnn/face_recognition_config.toml"
const MODEL_DATA_TOML_PATH = "../../../dados/webcamcnn/face_recognition_model_data.toml"

# Person data structure
struct PersonData
    name::String
    images::Vector{Array{Float32, 3}}
    label::Int
    is_incremental::Bool
end

# Enhanced image validation
function validate_image_file(filepath::String)
    try
        img = load(filepath)
        if size(img) == (0, 0)
            throw(ArgumentError("Empty or invalid image"))
        end
        return true
    catch e
        println("Invalid or corrupted image: $filepath ($e)")
        return false
    end
end

# Calculate feature dimensions for model architecture
function calculate_feature_dimensions(input_size::Tuple{Int, Int})
    h, w = input_size
    for i in 1:4  # 4 MaxPool layers
        h = div(h, 2)
        w = div(w, 2)
    end
    return h, w
end

# Extract model information for TOML
function extract_model_info_for_toml(model, person_names::Vector{String})
    model_info = Dict{String, Any}()
    
    model_info["model_summary"] = Dict(
        "total_layers" => length(model),
        "model_type" => "CNN_FaceRecognition",
        "input_shape" => collect(IMG_SIZE) .|> Int,
        "output_classes" => length(person_names),
        "created_at" => string(Dates.now())
    )
    
    # Layer information
    layer_info = []
    for (i, layer) in enumerate(model)
        layer_dict = Dict{String, Any}(
            "layer_number" => i,
            "layer_type" => string(typeof(layer)),
            "trainable" => true
        )
        
        try
            if isa(layer, Conv)
                if hasfield(typeof(layer), :weight) && layer.weight !== nothing
                    layer_dict["kernel_size"] = collect(size(layer.weight)[1:2]) .|> Int
                    layer_dict["input_channels"] = size(layer.weight)[3]
                    layer_dict["output_channels"] = size(layer.weight)[4]
                end
            elseif isa(layer, Dense)
                if hasfield(typeof(layer), :weight) && layer.weight !== nothing
                    layer_dict["input_size"] = size(layer.weight)[2]
                    layer_dict["output_size"] = size(layer.weight)[1]
                end
            elseif isa(layer, MaxPool)
                if hasfield(typeof(layer), :k)
                    layer_dict["pool_size"] = isa(layer.k, Tuple) ? collect(layer.k) : [layer.k]
                end
            elseif isa(layer, BatchNorm)
                if hasfield(typeof(layer), :β) && layer.β !== nothing
                    layer_dict["num_features"] = length(layer.β)
                end
            end
        catch e
            println("Warning: Could not extract info for layer $i: $e")
        end
        
        push!(layer_info, layer_dict)
    end
    model_info["layer_info"] = layer_info
    
    # Weight statistics
    total_params = 0
    weight_stats = Dict{String, Any}()
    
    for (i, layer) in enumerate(model)
        try
            if hasfield(typeof(layer), :weight) && layer.weight !== nothing
                w = layer.weight
                layer_params = length(w)
                total_params += layer_params
                
                weight_stats["layer_$(i)_weights"] = Dict(
                    "shape" => collect(size(w)) .|> Int,
                    "count" => layer_params,
                    "mean" => Float64(mean(w)),
                    "std" => Float64(std(w)),
                    "min" => Float64(minimum(w)),
                    "max" => Float64(maximum(w))
                )
            end
        catch e
            println("Warning: Could not extract weight stats for layer $i: $e")
        end
    end
    
    model_info["weights_summary"] = Dict(
        "total_parameters" => total_params,
        "layer_statistics" => weight_stats,
        "model_size_mb" => round(total_params * 4 / (1024^2), digits=2)
    )
    
    # Person mappings
    person_mappings = Dict{String, Any}()
    for (i, name) in enumerate(person_names)
        person_mappings[name] = i
    end
    model_info["person_mappings"] = person_mappings
    model_info["prediction_examples"] = []
    
    return model_info
end

# Save model data to TOML
function save_model_data_toml(model, person_names::Vector{String}, filepath::String)
    println("📝 Salvando dados do modelo em TOML...")
    
    try
        model_info = extract_model_info_for_toml(model, person_names)
        model_info["metadata"] = Dict(
            "format_version" => "1.0",
            "created_by" => "cnncheckin.jl v1.0",
            "description" => "CNN model for face recognition",
            "saved_at" => string(Dates.now())
        )
        
        open(filepath, "w") do io
            TOML.print(io, model_info)
        end
        println("✅ Dados do modelo salvos em: $filepath")
        return true
    catch e
        println("❌ Erro ao salvar dados do modelo: $e")
        return false
    end
end

# Load model data from TOML
function load_model_data_toml(filepath::String)
    if !isfile(filepath)
        println("⚠️ Arquivo de dados do modelo não encontrado: $filepath")
        return nothing
    end
    
    try
        model_data = TOML.parsefile(filepath)
        println("✅ Dados do modelo carregados de: $filepath")
        return model_data
    catch e
        println("❌ Erro ao carregar dados do modelo: $e")
        return nothing
    end
end

# Save configuration to TOML
function save_config(config::Dict, filepath::String)
    println("📁 Salvando configuração...")
    
    try
        config["metadata"]["last_saved"] = string(Dates.now())
        open(filepath, "w") do io
            TOML.print(io, config)
        end
        println("✅ Configuração salva em: $filepath")
        return true
    catch e
        println("❌ Erro ao salvar configuração: $e")
        return false
    end
end

# Load configuration from TOML
function load_config(filepath::String)
    println("📂 Carregando configuração...")
    
    if !isfile(filepath)
        println("⚠️ Arquivo de configuração não encontrado, criando padrão...")
        config = create_default_config()
        save_config(config, filepath)
        return config
    end
    
    try
        config = TOML.parsefile(filepath)
        println("✅ Configuração carregada de: $filepath")
        return config
    catch e
        println("❌ Erro ao carregar configuração: $e")
        println("Criando configuração padrão...")
        config = create_default_config()
        save_config(config, filepath)
        return config
    end
end

# Create default configuration
function create_default_config()
    return Dict(
        "model" => Dict(
            "img_width" => IMG_SIZE[1],
            "img_height" => IMG_SIZE[2],
            "num_classes" => 0,
            "model_architecture" => "CNN_v1",
            "augmentation_used" => true
        ),
        "training" => Dict(
            "batch_size" => BATCH_SIZE,
            "epochs" => PRETRAIN_EPOCHS,
            "learning_rate" => LEARNING_RATE,
            "epochs_trained" => 0,
            "final_accuracy" => 0.0,
            "best_epoch" => 0
        ),
        "data" => Dict(
            "person_names" => String[],
            "data_path" => TRAIN_DATA_PATH,
            "timestamp" => string(Dates.now())
        ),
        "metadata" => Dict(
            "created_by" => "cnncheckin.jl v1.0",
            "version" => "1.0",
            "description" => "Face recognition system configuration",
            "last_saved" => string(Dates.now())
        )
    )
end

# Validate configuration
function validate_config(config::Dict)
    required_sections = ["model", "training", "data", "metadata"]
    for section in required_sections
        if !haskey(config, section)
            error("Seção '$section' não encontrada na configuração")
        end
    end
    
    println("✅ Configuração é válida")
    return true
end

# Data augmentation
function augment_image(img_array::Array{Float32, 3})
    augmented = []
    push!(augmented, img_array)  # Original
    
    # Horizontal flip
    flipped = reverse(img_array, dims=2)
    push!(augmented, flipped)
    
    # Brightness variations
    bright = clamp.(img_array .* 1.1, 0.0f0, 1.0f0)
    push!(augmented, bright)
    
    dark = clamp.(img_array .* 0.9, 0.0f0, 1.0f0)
    push!(augmented, dark)
    
    # Light noise
    noise = img_array .+ 0.02f0 .* randn(Float32, size(img_array))
    noise_clamped = clamp.(noise, -2.0f0, 2.0f0)
    push!(augmented, noise_clamped)
    
    return augmented
end

# Preprocess image
function preprocess_image(img_path::String; augment::Bool = false)
    try
        img = load(img_path)
        
        # Handle different image formats
        if ndims(img) == 2
            img = Gray.(img)
            img = RGB.(img)
        elseif isa(img, Array) && eltype(img) <: RGBA
            img = RGB.(img)
        elseif isa(img, Array) && eltype(img) <: Gray
            img = RGB.(img)
        end
        
        img_resized = imresize(img, IMG_SIZE)
        img_array = Float32.(channelview(img_resized))
        img_array = permutedims(img_array, (2, 3, 1))
        img_array = Float32.(img_array)
        
        # Normalization
        μ = mean(img_array)
        σ = std(img_array)
        if σ > 1e-6
            img_array = (img_array .- μ) ./ σ
        end
        
        if augment
            return augment_image(img_array)
        else
            return [img_array]
        end
    catch e
        println("❌ Erro ao processar imagem $img_path: $e")
        return nothing
    end
end

# Extract person name from filename
function extract_person_name(filename::String)
    name_parts = split(splitext(filename)[1], "_")
    return name_parts[1]
end

# Function to create directory if it doesn't exist
function criar_diretorio(caminho)
    if !isdir(caminho)
        mkpath(caminho)
        println("Diretório criado: $caminho")
    end
end

# Export functions and constants
export PersonData, validate_image_file, calculate_feature_dimensions, 
       extract_model_info_for_toml, save_model_data_toml, load_model_data_toml, 
       save_config, load_config, create_default_config, 
       validate_config, preprocess_image, extract_person_name, augment_image,
       criar_diretorio,
       IMG_SIZE, BATCH_SIZE, PRETRAIN_EPOCHS, INCREMENTAL_EPOCHS, LEARNING_RATE, INCREMENTAL_LR,
       TRAIN_DATA_PATH, INCREMENTAL_DATA_PATH, AUTH_DATA_PATH, MODEL_PATH, 
       CONFIG_PATH, MODEL_DATA_TOML_PATH

end # module CNNCheckinCore