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

# Configurações globais
const IMG_SIZE = (128, 128)
const BATCH_SIZE = 8
const EPOCHS = 50
const LEARNING_RATE = 0.0001
const DATA_PATH = "../../../dados/fotos"
const MODEL_PATH = "face_recognition_model.jld2"
const CONFIG_PATH = "face_recognition_config.toml"
const MODEL_DATA_TOML_PATH = "face_recognition_model_data.toml"

# Estrutura para dados de uma pessoa
struct PersonData
    name::String
    images::Vector{Array{Float32, 3}}
    label::Int
end

# Estrutura para configurações do modelo
struct ModelConfig
    img_size::Tuple{Int, Int}
    num_classes::Int
    person_names::Vector{String}
    batch_size::Int
    epochs_trained::Int
    learning_rate::Float64
    final_accuracy::Float64
    timestamp::String
    data_path::String
    model_architecture::String
    augmentation_used::Bool
    early_stopping_patience::Int
    best_epoch::Int
end

# Função para validar integridade de arquivos de imagem
function validate_image_file(filepath::String)
    try
        img = load(filepath)
        if size(img) == (0, 0)
            throw(ArgumentError("Imagem vazia ou inválida"))
        end
        return true
    catch e
        println("⚠️ Imagem inválida ou corrompida: $filepath ($e)")
        println("💡 Sugestão: Re-encode com 'convert $filepath -strip $(filepath)_fixed.jpg'")
        return false
    end
end

# Função para extrair informações do modelo para TOML
function extract_model_info_for_toml(model, person_names::Vector{String})
    model_info = Dict{String, Any}()
    
    model_info["model_summary"] = Dict(
        "total_layers" => length(model),
        "model_type" => "CNN",
        "input_shape" => collect(IMG_SIZE) .|> Int,
        "output_classes" => length(person_names),
        "created_at" => string(Dates.now())
    )
    
    layer_info = []
    for (i, layer) in enumerate(model)
        layer_dict = Dict{String, Any}(
            "layer_number" => i,
            "layer_type" => string(typeof(layer)),
            "trainable" => true
        )
        
        if isa(layer, Conv)
            layer_dict["kernel_size"] = collect(size(layer.weight)[1:2]) .|> Int
            layer_dict["input_channels"] = size(layer.weight)[3]
            layer_dict["output_channels"] = size(layer.weight)[4]
            layer_dict["stride"] = collect(layer.stride) .|> Int
            layer_dict["pad"] = collect(layer.pad) .|> Int
        elseif isa(layer, Dense)
            layer_dict["input_size"] = size(layer.weight)[2]
            layer_dict["output_size"] = size(layer.weight)[1]
            layer_dict["has_bias"] = layer.bias !== false
        elseif isa(layer, MaxPool)
            layer_dict["pool_size"] = collect(layer.k) .|> Int
            layer_dict["stride"] = collect(layer.stride) .|> Int
        elseif isa(layer, BatchNorm)
            layer_dict["num_features"] = length(layer.β)
            layer_dict["eps"] = hasfield(typeof(layer), :epsilon) ? getfield(layer, :epsilon) : (hasfield(typeof(layer), :ε) ? getfield(layer, :ε) : "unknown")
            layer_dict["momentum"] = hasfield(typeof(layer), :momentum) ? layer.momentum : "unknown"
        elseif isa(layer, Dropout)
            layer_dict["dropout_rate"] = layer.p
        else
            layer_dict["description"] = string(layer)
        end
        
        push!(layer_info, layer_dict)
    end
    model_info["layer_info"] = layer_info
    
    total_params = 0
    weight_stats = Dict{String, Any}()
    
    for (i, layer) in enumerate(model)
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
        
        if hasfield(typeof(layer), :bias) && layer.bias !== nothing && layer.bias !== false
            b = layer.bias
            bias_params = length(b)
            total_params += bias_params
            
            weight_stats["layer_$(i)_bias"] = Dict(
                "shape" => collect(size(b)) .|> Int,
                "count" => bias_params,
                "mean" => Float64(mean(b)),
                "std" => Float64(std(b)),
                "min" => Float64(minimum(b)),
                "max" => Float64(maximum(b))
            )
        end
    end
    
    model_info["weights_summary"] = Dict(
        "total_parameters" => total_params,
        "layer_statistics" => weight_stats,
        "model_size_mb" => round(total_params * 4 / (1024^2), digits=2)
    )
    
    # CORREÇÃO: Mapeamento correto pessoa -> índice (começando de 1)
    person_mappings = Dict{String, Any}()
    for (i, name) in enumerate(person_names)
        person_mappings[name] = i  # Julia usa indexação começando em 1
    end
    model_info["person_mappings"] = person_mappings
    
    model_info["prediction_examples"] = []
    
    return model_info
end

# Função para salvar dados do modelo em TOML
function save_model_data_toml(model, person_names::Vector{String}, filepath::String)
    println("💾 Salvando dados do modelo em TOML...")
    
    try
        model_info = extract_model_info_for_toml(model, person_names)
        model_info["metadata"] = Dict(
            "format_version" => "1.0",
            "created_by" => "cnncheckin.jl v2.1",
            "description" => "Dados estruturais e estatísticas do modelo CNN",
            "saved_at" => string(Dates.now()),
            "companion_files" => [MODEL_PATH, CONFIG_PATH]
        )
        
        open(filepath, "w") do io
            TOML.print(io, model_info)
        end
        println("✅ Dados do modelo salvos em TOML: $filepath")
        return true
    catch e
        println("❌ Erro ao salvar dados do modelo em TOML: $e")
        return false
    end
end

# Função para carregar dados do modelo do TOML
function load_model_data_toml(filepath::String)
    println("📂 Carregando dados do modelo do TOML...")
    
    if !isfile(filepath)
        println("⚠️ Arquivo de dados do modelo não encontrado: $filepath")
        return nothing
    end
    
    try
        model_data = TOML.parsefile(filepath)
        println("✅ Dados do modelo carregados do TOML: $filepath")
        return model_data
    catch e
        println("❌ Erro ao carregar dados do modelo: $e")
        return nothing
    end
end

# Função para adicionar exemplo de predição ao TOML (CORRIGIDA)
function add_prediction_example_to_toml(image_path::String, predicted_person::String, 
                                       confidence::Float64, actual_person::String = "")
    model_data = load_model_data_toml(MODEL_DATA_TOML_PATH)
    if model_data === nothing
        return false
    end
    
    example = Dict{String, Any}(
        "timestamp" => string(Dates.now()),
        "image_path" => image_path,
        "predicted_person" => predicted_person,
        "confidence" => round(confidence, digits=4),
        "image_filename" => basename(image_path)
    )
    
    if !isempty(actual_person)
        example["actual_person"] = actual_person
        example["correct_prediction"] = predicted_person == actual_person
    end
    
    if !haskey(model_data, "prediction_examples")
        model_data["prediction_examples"] = Any[]
    end
    
    # Certificar que prediction_examples é um Array
    if !isa(model_data["prediction_examples"], Vector)
        model_data["prediction_examples"] = Any[]
    end
    
    push!(model_data["prediction_examples"], example)
    
    # Manter apenas os últimos 50 exemplos
    if length(model_data["prediction_examples"]) > 50
        model_data["prediction_examples"] = model_data["prediction_examples"][end-49:end]
    end
    
    try
        open(MODEL_DATA_TOML_PATH, "w") do io
            TOML.print(io, model_data)
        end
        return true
    catch e
        println("❌ Erro ao atualizar exemplos de predição: $e")
        return false
    end
end

# Função para salvar configuração em TOML
function save_config(config::Dict, filepath::String)
    println("💾 Salvando configuração em TOML...")
    
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

# Função para carregar configuração do TOML
function load_config(filepath::String)
    println("📂 Carregando configuração do TOML...")
    
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
        println("🔧 Criando configuração padrão...")
        config = create_default_config()
        save_config(config, filepath)
        return config
    end
end

# Função para criar configuração padrão
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
            "epochs_trained" => 0,
            "learning_rate" => LEARNING_RATE,
            "final_accuracy" => 0.0,
            "early_stopping_patience" => 10,
            "best_epoch" => 0
        ),
        "data" => Dict(
            "person_names" => String[],
            "data_path" => DATA_PATH,
            "timestamp" => string(Dates.now())
        ),
        "metadata" => Dict(
            "created_by" => "cnncheckin.jl",
            "version" => "2.1",
            "description" => "Configurações do modelo de reconhecimento facial",
            "last_saved" => string(Dates.now())
        )
    )
end

# Função para validar configuração
function validate_config(config::Dict)
    required_sections = ["model", "training", "data", "metadata"]
    for section in required_sections
        if !haskey(config, section)
            error("Seção '$section' não encontrada na configuração")
        end
    end
    model_section = config["model"]
    if !haskey(model_section, "num_classes") || model_section["num_classes"] <= 0
        error("num_classes deve ser maior que 0")
    end
    data_section = config["data"]
    if !haskey(data_section, "person_names") || isempty(data_section["person_names"])
        error("Lista de pessoas não pode estar vazia")
    end
    println("✅ Configuração válida")
    return true
end

# Função para aplicar data augmentation
function augment_image(img_array::Array{Float32, 3})
    augmented = []
    push!(augmented, img_array)  # Original
    flipped = reverse(img_array, dims=2)
    push!(augmented, flipped)
    bright = clamp.(img_array .* 1.2, 0.0f0, 1.0f0)
    dark = clamp.(img_array .* 0.8, 0.0f0, 1.0f0)
    push!(augmented, bright)
    push!(augmented, dark)
    noise = img_array .+ 0.05f0 .* randn(Float32, size(img_array))
    noise_clamped = clamp.(noise, -2.0f0, 2.0f0)
    push!(augmented, noise_clamped)
    return augmented
end

# Função para carregar e preprocessar uma imagem
function preprocess_image(img_path::String; augment::Bool = false)
    try
        img = load(img_path)
        if ndims(img) == 2
            img = Gray.(img)
            img = RGB.(img)
        elseif isa(img, Array) && eltype(img) <: RGBA
            img = RGB.(img)
        end
        img_resized = imresize(img, IMG_SIZE)
        img_array = Float32.(channelview(img_resized))
        img_array = permutedims(img_array, (2, 3, 1))
        img_array = Float32.(img_array)
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
        println("Erro ao processar imagem $img_path: $e")
        return nothing
    end
end

# Função para extrair nome da pessoa do nome do arquivo
function extract_person_name(filename::String)
    name_parts = split(splitext(filename)[1], "-")
    return name_parts[1]
end

# Exportar funções e estruturas
export PersonData, ModelConfig, validate_image_file, extract_model_info_for_toml, 
       save_model_data_toml, load_model_data_toml, add_prediction_example_to_toml,
       save_config, load_config, create_default_config, validate_config,
       preprocess_image, extract_person_name, IMG_SIZE, BATCH_SIZE, EPOCHS,
       LEARNING_RATE, DATA_PATH, MODEL_PATH, CONFIG_PATH, MODEL_DATA_TOML_PATH,
       augment_image

end # module CNNCheckinCore