# CNNCheckin - Sistema de Reconhecimento Facial
# Arquivo: CNNCheckin_Config.jl
# Configurações centralizadas do sistema

module CNNCheckinConfig

using TOML
using Dates

# ============================================================================
# CONSTANTES DO SISTEMA
# ============================================================================

# Dimensões de imagem
const IMG_WIDTH = 128
const IMG_HEIGHT = 128
const IMG_CHANNELS = 3

# Arquitetura da rede
const CONV_FILTERS = [64, 128, 256, 256]
const DENSE_UNITS = [512, 256]
const DROPOUT_RATES = [0.1, 0.1, 0.15, 0.15, 0.4, 0.3]

# Treinamento
const BATCH_SIZE = 8
const INITIAL_EPOCHS = 30
const INCREMENTAL_EPOCHS = 15
const LEARNING_RATE = 0.0001
const INCREMENTAL_LR = 0.00005
const EARLY_STOP_PATIENCE = 10
const MIN_IMPROVEMENT = 0.001

# Data augmentation
const AUGMENTATION_ENABLED = true
const FLIP_HORIZONTAL = true
const BRIGHTNESS_RANGE = 0.1
const NOISE_LEVEL = 0.02

# Caminhos padrão
const BASE_DIR = joinpath(@__DIR__, "..")
const DATA_DIR = joinpath(BASE_DIR, "dados")
const TRAIN_DIR = joinpath(DATA_DIR, "fotos_train")
const INCREMENTAL_DIR = joinpath(DATA_DIR, "fotos_new")
const TEST_DIR = joinpath(DATA_DIR, "fotos_auth")
const MODELS_DIR = joinpath(BASE_DIR, "src")
const LOGS_DIR = joinpath(BASE_DIR, "logs")

# Arquivos de modelo
const MODEL_FILE = joinpath(MODELS_DIR, "face_recognition_model.jld2")
const CONFIG_FILE = joinpath(MODELS_DIR, "face_recognition_config.toml")
const METADATA_FILE = joinpath(MODELS_DIR, "face_recognition_model_data.toml")

# Webcam
const DEFAULT_CAMERA = 0
const CAPTURE_WIDTH = 640
const CAPTURE_HEIGHT = 480
const PREVIEW_DURATION = 3
const CAPTURE_DELAY = 2

# Validação
const MIN_IMAGES_PER_PERSON = 5
const MIN_CONFIDENCE_THRESHOLD = 0.6
const HIGH_CONFIDENCE_THRESHOLD = 0.85

# Extensões suportadas
const IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]

# ============================================================================
# ESTRUTURAS DE DADOS
# ============================================================================

"""
Dados de uma pessoa no sistema
"""
struct PersonData
    name::String
    images::Vector{Array{Float32, 3}}
    label::Int
    is_new::Bool
end

"""
Configuração do modelo
"""
mutable struct ModelConfig
    num_classes::Int
    person_names::Vector{String}
    img_size::Tuple{Int, Int, Int}
    architecture::String
    created_at::DateTime
    last_trained::DateTime
    accuracy::Float64
    
    function ModelConfig(num_classes::Int, person_names::Vector{String})
        new(
            num_classes,
            person_names,
            (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS),
            "CNN_v2",
            now(),
            now(),
            0.0
        )
    end
end

"""
Resultados de treinamento
"""
struct TrainingResults
    train_losses::Vector{Float64}
    val_accuracies::Vector{Float64}
    best_accuracy::Float64
    best_epoch::Int
    total_epochs::Int
    duration_seconds::Float64
end

# ============================================================================
# FUNÇÕES DE CONFIGURAÇÃO
# ============================================================================

"""
Cria estrutura de diretórios necessária
"""
function setup_directories()
    dirs = [DATA_DIR, TRAIN_DIR, INCREMENTAL_DIR, TEST_DIR, MODELS_DIR, LOGS_DIR]
    
    for dir in dirs
        if !isdir(dir)
            mkpath(dir)
            @info "📁 Diretório criado: $dir"
        end
    end
end

"""
Salva configuração do modelo
"""
function save_model_config(config::ModelConfig)
    config_dict = Dict(
        "model" => Dict(
            "num_classes" => config.num_classes,
            "img_width" => IMG_WIDTH,
            "img_height" => IMG_HEIGHT,
            "img_channels" => IMG_CHANNELS,
            "architecture" => config.architecture
        ),
        "training" => Dict(
            "batch_size" => BATCH_SIZE,
            "learning_rate" => LEARNING_RATE,
            "accuracy" => config.accuracy
        ),
        "data" => Dict(
            "person_names" => config.person_names,
            "created_at" => string(config.created_at),
            "last_trained" => string(config.last_trained)
        ),
        "metadata" => Dict(
            "version" => "2.0",
            "system" => "CNNCheckin"
        )
    )
    
    open(CONFIG_FILE, "w") do io
        TOML.print(io, config_dict)
    end
    
    @info "💾 Configuração salva: $CONFIG_FILE"
end

"""
Carrega configuração do modelo
"""
function load_model_config()
    if !isfile(CONFIG_FILE)
        @warn "Arquivo de configuração não encontrado"
        return nothing
    end
    
    config_dict = TOML.parsefile(CONFIG_FILE)
    
    config = ModelConfig(
        config_dict["model"]["num_classes"],
        config_dict["data"]["person_names"]
    )
    
    config.accuracy = config_dict["training"]["accuracy"]
    config.created_at = DateTime(config_dict["data"]["created_at"])
    config.last_trained = DateTime(config_dict["data"]["last_trained"])
    
    return config
end

"""
Valida arquivo de imagem
"""
function is_valid_image(filepath::String)
    # Verifica extensão
    ext = lowercase(splitext(filepath)[2])
    if !(ext in IMAGE_EXTENSIONS)
        return false
    end
    
    # Verifica existência
    if !isfile(filepath)
        return false
    end
    
    # Verifica tamanho
    filesize = stat(filepath).size
    if filesize < 1024 || filesize > 10 * 1024 * 1024  # 1KB a 10MB
        return false
    end
    
    return true
end

"""
Extrai nome da pessoa do filename
"""
function extract_person_name(filename::String)
    base = splitext(filename)[1]
    # Remove números e underscores no final
    name = replace(base, r"[-_]\d+$" => "")
    return strip(name)
end

"""
Gera timestamp para arquivos
"""
function generate_timestamp()
    return Dates.format(now(), "yyyymmdd_HHMMSS")
end

# ============================================================================
# EXPORTAÇÕES
# ============================================================================

export IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS,
       CONV_FILTERS, DENSE_UNITS, DROPOUT_RATES,
       BATCH_SIZE, INITIAL_EPOCHS, INCREMENTAL_EPOCHS,
       LEARNING_RATE, INCREMENTAL_LR,
       EARLY_STOP_PATIENCE, MIN_IMPROVEMENT,
       AUGMENTATION_ENABLED, FLIP_HORIZONTAL, BRIGHTNESS_RANGE, NOISE_LEVEL,
       BASE_DIR, DATA_DIR, TRAIN_DIR, INCREMENTAL_DIR, TEST_DIR,
       MODELS_DIR, LOGS_DIR,
       MODEL_FILE, CONFIG_FILE, METADATA_FILE,
       DEFAULT_CAMERA, CAPTURE_WIDTH, CAPTURE_HEIGHT,
       PREVIEW_DURATION, CAPTURE_DELAY,
       MIN_IMAGES_PER_PERSON, MIN_CONFIDENCE_THRESHOLD, HIGH_CONFIDENCE_THRESHOLD,
       IMAGE_EXTENSIONS,
       PersonData, ModelConfig, TrainingResults,
       setup_directories, save_model_config, load_model_config,
       is_valid_image, extract_person_name, generate_timestamp

end # module