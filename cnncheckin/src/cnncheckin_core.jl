# projeto: cnncheckin
# file: cnncheckin/src/cnncheckin_core.jl
# descrição: Módulo central com funcionalidades compartilhadas

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

# ============================================================================
# CONSTANTES DE CONFIGURAÇÃO
# ============================================================================

const IMG_SIZE = (128, 128)
const BATCH_SIZE = 8
const PRETRAIN_EPOCHS = 30
const INCREMENTAL_EPOCHS = 15
const LEARNING_RATE = 0.0001
const INCREMENTAL_LR = 0.00005

# Caminhos de diretórios
const TRAIN_DATA_PATH = "../../../dados/fotos_train"
const INCREMENTAL_DATA_PATH = "../../../dados/fotos_new"
const AUTH_DATA_PATH = "../../../dados/fotos_auth"

# Arquivos do modelo
const MODEL_PATH = "face_recognition_model.jld2"
const CONFIG_PATH = "face_recognition_config.toml"
const MODEL_DATA_TOML_PATH = "face_recognition_model_data.toml"

# Extensões de imagem suportadas
const VALID_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp"]

# Configurações de validação
const MIN_FILE_SIZE_BYTES = 500
const MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50MB
const MIN_IMAGE_DIMENSION = 10

# ============================================================================
# ESTRUTURAS DE DADOS
# ============================================================================

"""
    PersonData

Estrutura para armazenar dados de uma pessoa no sistema.

# Campos
- `name::String`: Nome da pessoa
- `images::Vector{Array{Float32, 3}}`: Imagens processadas
- `label::Int`: Rótulo numérico da pessoa
- `is_incremental::Bool`: Se foi adicionada via aprendizado incremental
"""
struct PersonData
    name::String
    images::Vector{Array{Float32, 3}}
    label::Int
    is_incremental::Bool
end

# ============================================================================
# VALIDAÇÃO DE IMAGENS
# ============================================================================

"""
    validate_image_file(filepath::String) -> Bool

Valida se um arquivo é uma imagem válida para processamento.
"""
function validate_image_file(filepath::String)::Bool
    try
        # Verificar extensão
        file_ext = lowercase(splitext(filepath)[2])
        if !(file_ext in VALID_IMAGE_EXTENSIONS)
            @debug "Extensão não suportada: $filepath"
            return false
        end
        
        # Verificar existência do arquivo
        if !isfile(filepath)
            @warn "Arquivo não existe: $filepath"
            return false
        end
        
        # Verificar tamanho do arquivo
        filesize_bytes = stat(filepath).size
        if filesize_bytes < MIN_FILE_SIZE_BYTES
            @debug "Arquivo muito pequeno: $filepath"
            return false
        end
        
        if filesize_bytes > MAX_FILE_SIZE_BYTES
            @warn "Arquivo muito grande: $filepath"
            return false
        end
        
        # Tentar carregar a imagem
        img = load(filepath)
        
        # Verificar dimensões
        if ndims(img) < 2
            @debug "Dimensões inválidas: $filepath"
            return false
        end
        
        img_size = size(img)
        if length(img_size) >= 2 && (img_size[1] < MIN_IMAGE_DIMENSION || img_size[2] < MIN_IMAGE_DIMENSION)
            @debug "Imagem muito pequena: $filepath"
            return false
        end
        
        return true
        
    catch e
        @error "Erro ao validar arquivo: $filepath" exception=(e, catch_backtrace())
        return false
    end
end

# ============================================================================
# PROCESSAMENTO DE IMAGENS
# ============================================================================

"""
    normalize_image(img_array::Array{Float32, 3}) -> Array{Float32, 3}

Normaliza uma imagem usando padronização (z-score).
"""
function normalize_image(img_array::Array{Float32, 3})::Array{Float32, 3}
    μ = mean(img_array)
    σ = std(img_array)
    
    if σ > 1e-6
        return (img_array .- μ) ./ σ
    else
        return img_array .- μ
    end
end

"""
    convert_to_rgb(img) -> Matrix{RGB}

Converte imagem para formato RGB, independente do formato original.
"""
function convert_to_rgb(img)
    if ndims(img) == 2 || eltype(img) <: Gray
        return RGB.(Gray.(img))
    elseif eltype(img) <: RGBA
        return RGB.(img)
    else
        return RGB.(img)
    end
end

"""
    augment_image(img_array::Array{Float32, 3}) -> Vector{Array{Float32, 3}}

Aplica técnicas de data augmentation em uma imagem.

# Transformações aplicadas
- Imagem original
- Flip horizontal
- Variação de brilho (+10%)
- Variação de brilho (-10%)
- Adição de ruído gaussiano leve
"""
function augment_image(img_array::Array{Float32, 3})::Vector{Array{Float32, 3}}
    augmented = Array{Float32, 3}[]
    
    # Original
    push!(augmented, img_array)
    
    # Flip horizontal
    push!(augmented, reverse(img_array, dims=2))
    
    # Variações de brilho
    push!(augmented, clamp.(img_array .* 1.1f0, -2.0f0, 2.0f0))
    push!(augmented, clamp.(img_array .* 0.9f0, -2.0f0, 2.0f0))
    
    # Ruído gaussiano leve
    noise = img_array .+ 0.02f0 .* randn(Float32, size(img_array))
    push!(augmented, clamp.(noise, -2.0f0, 2.0f0))
    
    return augmented
end

"""
    preprocess_image(img_path::String; augment::Bool = false) -> Union{Vector{Array{Float32, 3}}, Nothing}

Carrega e preprocessa uma imagem para o modelo.

# Argumentos
- `img_path::String`: Caminho para a imagem
- `augment::Bool`: Se deve aplicar data augmentation

# Retorna
- `Vector{Array{Float32, 3}}`: Vetor de imagens processadas (1 ou mais se augment=true)
- `Nothing`: Se houver erro no processamento
"""
function preprocess_image(img_path::String; augment::Bool = false)
    try
        # Carregar imagem
        img = load(img_path)
        
        # Converter para RGB
        img = convert_to_rgb(img)
        
        # Redimensionar
        img_resized = imresize(img, IMG_SIZE)
        
        # Converter para array Float32
        img_array = Float32.(channelview(img_resized))
        img_array = permutedims(img_array, (2, 3, 1))
        
        # Normalizar
        img_array = normalize_image(img_array)
        
        # Aplicar augmentation se solicitado
        if augment
            return augment_image(img_array)
        else
            return [img_array]
        end
        
    catch e
        @error "Erro ao processar imagem: $img_path" exception=(e, catch_backtrace())
        return nothing
    end
end

"""
    extract_person_name(filename::String) -> String

Extrai o nome da pessoa a partir do nome do arquivo.
Espera formato: nome-numero.extensao
"""
function extract_person_name(filename::String)::String
    base_name = splitext(filename)[1]
    name_parts = split(base_name, "-")
    return name_parts[1]
end

# ============================================================================
# GERENCIAMENTO DE CONFIGURAÇÃO
# ============================================================================

"""
    create_default_config() -> Dict

Cria uma configuração padrão para o sistema.
"""
function create_default_config()::Dict
    return Dict(
        "model" => Dict(
            "img_width" => IMG_SIZE[1],
            "img_height" => IMG_SIZE[2],
            "num_classes" => 0,
            "model_architecture" => "CNN_FaceRecognition_v1",
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
            "created_by" => "CNNCheckin v2.0",
            "version" => "2.0",
            "description" => "Sistema de reconhecimento facial com aprendizado incremental",
            "last_saved" => string(Dates.now())
        )
    )
end

"""
    validate_config(config::Dict) -> Bool

Valida se uma configuração possui todas as seções necessárias.
"""
function validate_config(config::Dict)::Bool
    required_sections = ["model", "training", "data", "metadata"]
    
    for section in required_sections
        if !haskey(config, section)
            throw(ArgumentError("Seção '$section' não encontrada na configuração"))
        end
    end
    
    @info "✅ Configuração válida"
    return true
end

"""
    save_config(config::Dict, filepath::String) -> Bool

Salva a configuração em arquivo TOML.
"""
function save_config(config::Dict, filepath::String)::Bool
    @info "💾 Salvando configuração..."
    
    try
        config["metadata"]["last_saved"] = string(Dates.now())
        
        open(filepath, "w") do io
            TOML.print(io, config)
        end
        
        @info "✅ Configuração salva: $filepath"
        return true
        
    catch e
        @error "Erro ao salvar configuração" exception=(e, catch_backtrace())
        return false
    end
end

"""
    load_config(filepath::String) -> Dict

Carrega a configuração de arquivo TOML.
"""
function load_config(filepath::String)::Dict
    @info "📂 Carregando configuração..."
    
    if !isfile(filepath)
        @warn "Arquivo de configuração não encontrado, criando padrão..."
        config = create_default_config()
        save_config(config, filepath)
        return config
    end
    
    try
        config = TOML.parsefile(filepath)
        @info "✅ Configuração carregada: $filepath"
        return config
        
    catch e
        @error "Erro ao carregar configuração" exception=(e, catch_backtrace())
        @info "Criando configuração padrão..."
        config = create_default_config()
        save_config(config, filepath)
        return config
    end
end

# ============================================================================
# GERENCIAMENTO DE METADADOS DO MODELO (TOML)
# ============================================================================

"""
    extract_model_info_for_toml(model, person_names::Vector{String}) -> Dict

Extrai informações detalhadas do modelo para salvar em TOML.
"""
function extract_model_info_for_toml(model, person_names::Vector{String})::Dict
    model_info = Dict{String, Any}()
    
    # Informações gerais
    model_info["model_summary"] = Dict(
        "total_layers" => length(model),
        "model_type" => "CNN_FaceRecognition",
        "input_shape" => collect(IMG_SIZE),
        "output_classes" => length(person_names),
        "created_at" => string(Dates.now())
    )
    
    # Informações das camadas
    layer_info = []
    for (i, layer) in enumerate(model)
        layer_dict = Dict{String, Any}(
            "layer_number" => i,
            "layer_type" => string(typeof(layer)),
            "trainable" => true
        )
        
        # Extrair informações específicas por tipo de camada
        try
            if isa(layer, Conv) && hasfield(typeof(layer), :weight) && layer.weight !== nothing
                layer_dict["kernel_size"] = collect(size(layer.weight)[1:2])
                layer_dict["input_channels"] = size(layer.weight)[3]
                layer_dict["output_channels"] = size(layer.weight)[4]
                
            elseif isa(layer, Dense) && hasfield(typeof(layer), :weight) && layer.weight !== nothing
                layer_dict["input_size"] = size(layer.weight)[2]
                layer_dict["output_size"] = size(layer.weight)[1]
                
            elseif isa(layer, MaxPool) && hasfield(typeof(layer), :k)
                layer_dict["pool_size"] = isa(layer.k, Tuple) ? collect(layer.k) : [layer.k]
                
            elseif isa(layer, BatchNorm) && hasfield(typeof(layer), :β) && layer.β !== nothing
                layer_dict["num_features"] = length(layer.β)
            end
        catch e
            @debug "Não foi possível extrair info da camada $i" exception=e
        end
        
        push!(layer_info, layer_dict)
    end
    model_info["layer_info"] = layer_info
    
    # Estatísticas dos pesos
    total_params = 0
    weight_stats = Dict{String, Any}()
    
    for (i, layer) in enumerate(model)
        try
            if hasfield(typeof(layer), :weight) && layer.weight !== nothing
                w = layer.weight
                layer_params = length(w)
                total_params += layer_params
                
                weight_stats["layer_$(i)_weights"] = Dict(
                    "shape" => collect(size(w)),
                    "count" => layer_params,
                    "mean" => Float64(mean(w)),
                    "std" => Float64(std(w)),
                    "min" => Float64(minimum(w)),
                    "max" => Float64(maximum(w))
                )
            end
        catch e
            @debug "Não foi possível extrair estatísticas da camada $i" exception=e
        end
    end
    
    model_info["weights_summary"] = Dict(
        "total_parameters" => total_params,
        "layer_statistics" => weight_stats,
        "model_size_mb" => round(total_params * 4 / (1024^2), digits=2)
    )
    
    # Mapeamento de pessoas
    person_mappings = Dict{String, Int}()
    for (i, name) in enumerate(person_names)
        person_mappings[name] = i
    end
    model_info["person_mappings"] = person_mappings
    model_info["prediction_examples"] = []
    
    return model_info
end

"""
    save_model_data_toml(model, person_names::Vector{String}, filepath::String) -> Bool

Salva metadados do modelo em arquivo TOML.
"""
function save_model_data_toml(model, person_names::Vector{String}, filepath::String)::Bool
    @info "💾 Salvando metadados do modelo..."
    
    try
        model_info = extract_model_info_for_toml(model, person_names)
        model_info["metadata"] = Dict(
            "format_version" => "2.0",
            "created_by" => "CNNCheckin v2.0",
            "description" => "Metadados do modelo CNN para reconhecimento facial",
            "saved_at" => string(Dates.now())
        )
        
        open(filepath, "w") do io
            TOML.print(io, model_info)
        end
        
        @info "✅ Metadados salvos: $filepath"
        return true
        
    catch e
        @error "Erro ao salvar metadados" exception=(e, catch_backtrace())
        return false
    end
end

"""
    load_model_data_toml(filepath::String) -> Union{Dict, Nothing}

Carrega metadados do modelo de arquivo TOML.
"""
function load_model_data_toml(filepath::String)
    if !isfile(filepath)
        @warn "Arquivo de metadados não encontrado: $filepath"
        return nothing
    end
    
    try
        model_data = TOML.parsefile(filepath)
        @info "✅ Metadados carregados: $filepath"
        return model_data
        
    catch e
        @error "Erro ao carregar metadados" exception=(e, catch_backtrace())
        return nothing
    end
end

"""
    add_prediction_example_to_toml(image_path::String, predicted_person::String, 
                                   confidence::Float64, actual_person::String="") -> Bool

Adiciona exemplo de predição aos metadados do modelo.
"""
function add_prediction_example_to_toml(image_path::String, predicted_person::String, 
                                       confidence::Float64, actual_person::String="")::Bool
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
        @error "Erro ao atualizar exemplos de predição" exception=(e, catch_backtrace())
        return false
    end
end

# ============================================================================
# EXPORTAÇÕES
# ============================================================================

export PersonData,
       validate_image_file,
       normalize_image,
       convert_to_rgb,
       augment_image,
       preprocess_image,
       extract_person_name,
       create_default_config,
       validate_config,
       save_config,
       load_config,
       extract_model_info_for_toml,
       save_model_data_toml,
       load_model_data_toml,
       add_prediction_example_to_toml,
       IMG_SIZE,
       BATCH_SIZE,
       PRETRAIN_EPOCHS,
       INCREMENTAL_EPOCHS,
       LEARNING_RATE,
       INCREMENTAL_LR,
       TRAIN_DATA_PATH,
       INCREMENTAL_DATA_PATH,
       AUTH_DATA_PATH,
       MODEL_PATH,
       CONFIG_PATH,
       MODEL_DATA_TOML_PATH,
       VALID_IMAGE_EXTENSIONS

end  # module CNNCheckinCore