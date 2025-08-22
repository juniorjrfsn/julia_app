# module cnncheckin 
module cnncheckin

# projeto: cnncheckin
# file: cnncheckin/src/cnncheckin.jl
# Sistema de Reconhecimento Facial para Check-in
# Autor: Assistente IA
# Descrição: Treina uma CNN para reconhecimento de faces

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

# Configurações globais - AJUSTADAS PARA REDUZIR OVERFITTING
const IMG_SIZE = (128, 128)  # Aumentado para melhor qualidade
const BATCH_SIZE = 8  # Reduzido para datasets pequenos
const EPOCHS = 50  # Reduzido para evitar overfitting
const LEARNING_RATE = 0.0001  # Reduzido para aprendizado mais suave
const DATA_PATH = "../../../dados/fotos"
const MODEL_PATH = "face_recognition_model.jld2"
const CONFIG_PATH = "face_recognition_config.toml"

# Estrutura para armazenar dados de uma pessoa
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
            "version" => "2.0",
            "description" => "Configurações do modelo de reconhecimento facial"
        )
    )
end

# Função para salvar configuração em TOML
function save_config(config::Dict, filepath::String)
    println("💾 Salvando configuração em TOML...")
    
    try
        # Adicionar timestamp de salvamento
        config["metadata"]["last_saved"] = string(Dates.now())
        
        # Salvar em formato TOML
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
        println("⚠️  Arquivo de configuração não encontrado, criando padrão...")
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
        println("📝 Criando configuração padrão...")
        config = create_default_config()
        save_config(config, filepath)
        return config
    end
end

# Função para validar configuração
function validate_config(config::Dict)
    required_sections = ["model", "training", "data", "metadata"]
    
    for section in required_sections
        if !haskey(config, section)
            error("Seção '$section' não encontrada na configuração")
        end
    end
    
    # Validar campos essenciais
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
    
    # Flip horizontal
    flipped = reverse(img_array, dims=2)
    push!(augmented, flipped)
    
    # Ajuste de brilho
    bright = clamp.(img_array .* 1.2, 0.0f0, 1.0f0)
    dark = clamp.(img_array .* 0.8, 0.0f0, 1.0f0)
    push!(augmented, bright)
    push!(augmented, dark)
    
    # Adicionar ruído
    noise = img_array .+ 0.05f0 .* randn(Float32, size(img_array))
    noise_clamped = clamp.(noise, -2.0f0, 2.0f0)  # Para z-score normalization
    push!(augmented, noise_clamped)
    
    return augmented
end

# Função para carregar e preprocessar uma imagem - MELHORADA
function preprocess_image(img_path::String; augment::Bool = false)
    try
        # Carregar imagem
        img = load(img_path)
        
        # Converter para RGB se necessário
        if ndims(img) == 2
            img = Gray.(img)
            img = RGB.(img)
        elseif isa(img, Array) && eltype(img) <: RGBA
            # RGBA -> RGB
            img = RGB.(img)
        end
        
        # Redimensionar para tamanho padrão
        img_resized = imresize(img, IMG_SIZE)
        
        # Converter para array Float32
        img_array = Float32.(channelview(img_resized))
        
        # Reordenar dimensões para (height, width, channels)
        img_array = permutedims(img_array, (2, 3, 1))
        
        # Normalização mais simples (0-1) primeiro, depois Z-score
        img_array = Float32.(img_array)
        
        # Normalização Z-score apenas se não for constante
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
    # Remove extensão e pega parte antes do traço
    name_parts = split(splitext(filename)[1], "-")
    return name_parts[1]
end

# Função para carregar dados das imagens - MELHORADA COM DATA AUGMENTATION
function load_face_data(data_path::String; use_augmentation::Bool = true)
    println("📄 Carregando dados das imagens...")
    
    if !isdir(data_path)
        error("Diretório $data_path não encontrado!")
    end
    
    # Dicionário para armazenar imagens por pessoa
    person_images = Dict{String, Vector{Array{Float32, 3}}}()
    
    # Listar todos os arquivos de imagem
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    
    for filename in readdir(data_path)
        file_ext = lowercase(splitext(filename)[2])
        
        if file_ext in image_extensions
            person_name = extract_person_name(filename)
            img_path = joinpath(data_path, filename)
            
            # Preprocessar imagem com ou sem augmentation
            img_arrays = preprocess_image(img_path; augment=use_augmentation)
            
            if img_arrays !== nothing
                if !haskey(person_images, person_name)
                    person_images[person_name] = Vector{Array{Float32, 3}}()
                end
                
                for img_array in img_arrays
                    push!(person_images[person_name], img_array)
                end
                
                total_imgs = use_augmentation ? length(img_arrays) : 1
                println("✅ Carregada: $filename -> $person_name ($total_imgs variações)")
            else
                println("⚠️  Falha ao carregar: $filename")
            end
        end
    end
    
    # Converter para formato adequado
    people_data = Vector{PersonData}()
    person_names = sort(collect(keys(person_images)))
    
    for (idx, person_name) in enumerate(person_names)
        images = person_images[person_name]
        if length(images) > 0
            push!(people_data, PersonData(person_name, images, idx))
            println("👤 Pessoa: $person_name - $(length(images)) imagens")
        end
    end
    
    return people_data, person_names
end

# Função para criar datasets balanceados - MELHORADA
function create_datasets(people_data::Vector{PersonData}, split_ratio::Float64 = 0.8)
    println("📊 Criando datasets de treino e validação...")
    
    train_images = Vector{Array{Float32, 3}}()
    train_labels = Vector{Int}()
    val_images = Vector{Array{Float32, 3}}()
    val_labels = Vector{Int}()
    
    # Dividir por pessoa para manter balanceamento
    for person in people_data
        n_imgs = length(person.images)
        n_train = max(1, Int(floor(n_imgs * split_ratio)))
        
        # Embaralhar imagens desta pessoa
        indices = randperm(n_imgs)
        
        # Treino
        for i in 1:n_train
            push!(train_images, person.images[indices[i]])
            push!(train_labels, person.label)
        end
        
        # Validação
        for i in (n_train+1):n_imgs
            push!(val_images, person.images[indices[i]])
            push!(val_labels, person.label)
        end
    end
    
    println("📈 Dataset criado:")
    println("   - Treino: $(length(train_images)) imagens")
    println("   - Validação: $(length(val_images)) imagens")
    
    return (train_images, train_labels), (val_images, val_labels)
end

# Função para criar batches
function create_batches(images, labels, batch_size)
    batches = []
    n_samples = length(images)
    
    if n_samples == 0
        return batches
    end
    
    # Determinar o número máximo de classes para one-hot encoding
    max_label = maximum(labels)
    
    for i in 1:batch_size:n_samples
        end_idx = min(i + batch_size - 1, n_samples)
        batch_images = images[i:end_idx]
        batch_labels = labels[i:end_idx]
        
        # Converter para tensor 4D (height, width, channels, batch_size)
        batch_tensor = cat(batch_images..., dims=4)
        
        # One-hot encoding dos labels
        batch_labels_onehot = Flux.onehotbatch(batch_labels, 1:max_label)
        
        push!(batches, (batch_tensor, batch_labels_onehot))
    end
    
    return batches
end

# Arquitetura da CNN - MELHORADA CONTRA OVERFITTING
function create_cnn_model(num_classes::Int, input_size::Tuple{Int, Int} = IMG_SIZE)
    # Calcular o tamanho após as convoluções e pooling
    # 128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8
    final_size = div(div(div(div(input_size[1], 2), 2), 2), 2)  # 8
    final_features = 128 * final_size * final_size  # 128 * 8 * 8 = 8192
    
    return Chain(
        # Primeira camada convolucional - mais conservadora
        Conv((3, 3), 3 => 32, relu, pad=1),
        BatchNorm(32),
        Dropout(0.1),  # Dropout leve nas convoluções
        MaxPool((2, 2)),
        
        # Segunda camada convolucional
        Conv((3, 3), 32 => 64, relu, pad=1),
        BatchNorm(64),
        Dropout(0.1),
        MaxPool((2, 2)),
        
        # Terceira camada convolucional
        Conv((3, 3), 64 => 128, relu, pad=1),
        BatchNorm(128),
        Dropout(0.2),
        MaxPool((2, 2)),
        
        # Quarta camada convolucional
        Conv((3, 3), 128 => 128, relu, pad=1),  # Não aumenta canais
        BatchNorm(128),
        Dropout(0.2),
        MaxPool((2, 2)),
        
        # Flatten e camadas densas - MUITO mais conservadoras
        Flux.flatten,
        Dense(final_features, 256, relu),  # Reduzido drasticamente
        Dropout(0.5),
        Dense(256, 64, relu),  # Camada intermediária menor
        Dropout(0.4),
        Dense(64, num_classes)  # Sem softmax aqui, aplicado na loss
    )
end

# Função para calcular acurácia
function accuracy(model, data_loader)
    correct = 0
    total = 0
    
    for (x, y) in data_loader
        ŷ = softmax(model(x))  # Aplicar softmax para predição
        pred = Flux.onecold(ŷ)
        true_labels = Flux.onecold(y)
        correct += sum(pred .== true_labels)
        total += length(true_labels)
    end
    
    return total > 0 ? correct / total : 0.0
end

# Função de treinamento - MELHORADA COM EARLY STOPPING
function train_model(model, train_data, val_data, epochs, learning_rate)
    println("🚀 Iniciando treinamento...")
    
    # Configurar otimizador com weight decay
    optimizer = ADAM(learning_rate, (0.9, 0.999), 1e-8)  # Parâmetros mais conservadores
    opt_state = Flux.setup(optimizer, model)
    
    # Histórico de treinamento
    train_losses = Float64[]
    val_accuracies = Float64[]
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    patience_limit = 10  # Early stopping
    
    # Loop de treinamento
    for epoch in 1:epochs
        epoch_loss = 0.0
        num_batches = 0
        
        # Treinar por um epoch
        for (x, y) in train_data
            loss, grads = Flux.withgradient(model) do m
                ŷ = m(x)
                # Cross entropy com logits (mais estável numericamente)
                Flux.logitcrossentropy(ŷ, y)
            end
            
            # Atualizar parâmetros
            Flux.update!(opt_state, model, grads[1])
            
            epoch_loss += loss
            num_batches += 1
        end
        
        # Calcular métricas
        avg_loss = num_batches > 0 ? epoch_loss / num_batches : 0.0
        val_acc = accuracy(model, val_data)
        
        push!(train_losses, avg_loss)
        push!(val_accuracies, val_acc)
        
        # Early stopping logic
        if val_acc > best_val_acc
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
        else
            patience_counter += 1
        end
        
        # Exibir progresso
        if epoch % 5 == 0 || epoch == 1
            println("Epoch $epoch/$epochs - Loss: $(round(avg_loss, digits=4)) - Val Acc: $(round(val_acc*100, digits=2))% - Best: $(round(best_val_acc*100, digits=2))% (Epoch $best_epoch)")
        end
        
        # Parar se não melhorar
        if patience_counter >= patience_limit
            println("🛑 Early stopping - sem melhoria por $patience_limit epochs")
            break
        end
        
        # Parar se accuracy boa o suficiente (mas não perfeita)
        if val_acc >= 0.85 && val_acc < 0.98  # Evita overfitting
            println("🎯 Acurácia boa alcançada sem overfitting!")
            break
        end
    end
    
    return train_losses, val_accuracies, best_val_acc, best_epoch
end

# Função para salvar modelo e configuração
function save_model_and_config(model, person_names, model_filepath::String, config_filepath::String, training_info::Dict)
    println("💾 Salvando modelo e configuração...")
    
    # 1. Salvar modelo em JLD2 (pesos neurais)
    model_data = Dict(
        "model_state" => model,
        "timestamp" => string(Dates.now())
    )
    
    try
        jldsave(model_filepath; model_data=model_data)
        println("✅ Pesos do modelo salvos em: $model_filepath")
    catch e
        println("❌ Erro ao salvar modelo: $e")
        return false
    end
    
    # 2. Criar e salvar configuração em TOML
    config = create_default_config()
    
    # Atualizar com informações do treinamento atual
    config["model"]["num_classes"] = length(person_names)
    config["model"]["augmentation_used"] = training_info["augmentation_used"]
    
    config["training"]["epochs_trained"] = training_info["epochs_trained"]
    config["training"]["final_accuracy"] = training_info["final_accuracy"]
    config["training"]["best_epoch"] = training_info["best_epoch"]
    
    config["data"]["person_names"] = person_names
    config["data"]["timestamp"] = string(Dates.now())
    
    # Adicionar estatísticas de treinamento
    config["training_stats"] = Dict(
        "total_training_images" => training_info["total_training_images"],
        "total_validation_images" => training_info["total_validation_images"],
        "early_stopped" => training_info["early_stopped"],
        "training_duration_minutes" => get(training_info, "duration_minutes", 0.0)
    )
    
    return save_config(config, config_filepath)
end

# Função para carregar modelo e configuração
function load_model_and_config(model_filepath::String, config_filepath::String)
    println("📂 Carregando modelo e configuração...")
    
    # 1. Carregar configuração TOML
    config = load_config(config_filepath)
    validate_config(config)
    
    # 2. Verificar se o modelo existe
    if !isfile(model_filepath)
        error("Arquivo do modelo não encontrado: $model_filepath")
    end
    
    # 3. Carregar pesos do modelo
    try
        data = load(model_filepath)
        model_data = data["model_data"]
        model_state = model_data["model_state"]
        
        # Extrair informações da configuração
        person_names = config["data"]["person_names"]
        num_classes = config["model"]["num_classes"]
        
        println("✅ Modelo e configuração carregados com sucesso!")
        println("📊 Informações do modelo:")
        println("   - Classes: $num_classes")
        println("   - Pessoas: $(join(person_names, ", "))")
        println("   - Acurácia: $(round(config["training"]["final_accuracy"]*100, digits=2))%")
        println("   - Criado em: $(config["data"]["timestamp"])")
        
        return model_state, person_names, config
    catch e
        error("Erro ao carregar modelo: $e")
    end
end

# Função para fazer predição em uma nova imagem - MELHORADA
function predict_person(model, person_names, img_path::String)
    # Preprocessar imagem (sem augmentation para predição)
    img_arrays = preprocess_image(img_path; augment=false)
    
    if img_arrays === nothing || length(img_arrays) == 0
        return nothing, 0.0
    end
    
    img_array = img_arrays[1]  # Pegar apenas a original
    
    # Adicionar dimensão de batch
    img_tensor = reshape(img_array, size(img_array)..., 1)
    
    # Fazer predição
    logits = model(img_tensor)
    prediction = softmax(logits)  # Aplicar softmax
    
    # Obter classe e confiança
    pred_probs = vec(prediction)
    pred_class = argmax(pred_probs)
    confidence = pred_probs[pred_class]
    
    person_name = person_names[pred_class]
    
    return person_name, confidence
end

# Função de treinamento (comando --treino)
function train_command()
    println("🎯 Sistema de Reconhecimento Facial - Modo Treinamento")
    
    start_time = time()
    
    try
        # 1. Carregar dados com data augmentation
        people_data, person_names = load_face_data(DATA_PATH; use_augmentation=true)
        
        if length(people_data) == 0
            error("Nenhum dado encontrado!")
        end
        
        num_classes = length(person_names)
        println("👥 Total de pessoas: $num_classes")
        
        # Verificar se há dados suficientes
        total_images = sum(length(person.images) for person in people_data)
        println("📊 Total de imagens (com augmentation): $total_images")
        
        # 2. Criar datasets balanceados
        (train_images, train_labels), (val_images, val_labels) = create_datasets(people_data)
        
        # 3. Criar batches
        train_batches = create_batches(train_images, train_labels, BATCH_SIZE)
        val_batches = create_batches(val_images, val_labels, BATCH_SIZE)
        
        if length(train_batches) == 0
            error("Não foi possível criar batches de treinamento!")
        end
        
        # 4. Criar modelo
        println("🧠 Criando modelo CNN...")
        model = create_cnn_model(num_classes)
        
        # 5. Treinar modelo
        train_losses, val_accuracies, best_val_acc, best_epoch = train_model(model, train_batches, val_batches, EPOCHS, LEARNING_RATE)
        
        # 6. Calcular duração do treinamento
        end_time = time()
        duration_minutes = (end_time - start_time) / 60
        
        # 7. Preparar informações de treinamento
        training_info = Dict(
            "epochs_trained" => length(val_accuracies),
            "final_accuracy" => best_val_acc,
            "best_epoch" => best_epoch,
            "total_training_images" => length(train_images),
            "total_validation_images" => length(val_images),
            "augmentation_used" => true,
            "early_stopped" => length(val_accuracies) < EPOCHS,
            "duration_minutes" => duration_minutes
        )
        
        # 8. Exibir resultados finais
        println("🎉 Treinamento concluído!")
        println("📊 Resultados:")
        println("   - Melhor acurácia: $(round(best_val_acc*100, digits=2))% (Epoch $best_epoch)")
        println("   - Epochs treinados: $(training_info["epochs_trained"])/$EPOCHS")
        println("   - Duração: $(round(duration_minutes, digits=1)) minutos")
        println("   - Early stopping: $(training_info["early_stopped"] ? "Sim" : "Não")")
        
        # 9. Salvar modelo e configuração
        success = save_model_and_config(model, person_names, MODEL_PATH, CONFIG_PATH, training_info)
        
        if success
            println("✅ Sistema treinado e salvo com sucesso!")
            println("📄 Configuração salva em: $CONFIG_PATH")
            println("🧠 Pesos do modelo salvos em: $MODEL_PATH")
        else
            println("⚠️  Modelo treinado mas houve erro ao salvar configuração")
        end
        
        return success
        
    catch e
        println("❌ Erro durante treinamento: $e")
        return false
    end
end

# Função de identificação (comando --identifica)
function identify_command(image_path::String)
    println("🔍 Sistema de Reconhecimento Facial - Modo Identificação")
    println("📸 Analisando imagem: $image_path")
    
    try
        # Verificar se o arquivo existe
        if !isfile(image_path)
            error("Arquivo de imagem não encontrado: $image_path")
        end
        
        # Verificar se os arquivos do modelo existem
        if !isfile(MODEL_PATH)
            error("Modelo não encontrado! Execute primeiro: julia cnncheckin.jl --treino")
        end
        
        if !isfile(CONFIG_PATH)
            error("Configuração não encontrada! Execute primeiro: julia cnncheckin.jl --treino")
        end
        
        # Carregar modelo e configuração
        model, person_names, config = load_model_and_config(MODEL_PATH, CONFIG_PATH)
        
        # Fazer predição
        person_name, confidence = predict_person(model, person_names, image_path)
        
        if person_name === nothing
            println("❌ Não foi possível processar a imagem")
            return false
        end
        
        # Exibir resultado
        println("🎯 Resultado da Identificação:")
        println("   👤 Pessoa: $person_name")
        println("   📈 Confiança: $(round(confidence*100, digits=2))%")
        
        # Exibir informações adicionais do modelo
        println("\n📊 Informações do Modelo:")
        println("   🧠 Acurácia do modelo: $(round(config["training"]["final_accuracy"]*100, digits=2))%")
        println("   📅 Treinado em: $(config["data"]["timestamp"])")
        println("   🎓 Melhor epoch: $(config["training"]["best_epoch"])")
        
        # Avaliar confiança de forma mais conservadora
        if confidence >= 0.7
            println("✅ Identificação com alta confiança")
        elseif confidence >= 0.5
            println("⚠️  Identificação com confiança média")
        else
            println("❓ Identificação com baixa confiança - pode ser pessoa desconhecida")
        end
        
        return true
        
    catch e
        println("❌ Erro durante identificação: $e")
        return false
    end
end

# Função para exibir informações do modelo (comando --info)
function info_command()
    println("📋 Informações do Modelo de Reconhecimento Facial")
    
    if !isfile(CONFIG_PATH)
        println("❌ Configuração não encontrada. Execute primeiro: julia cnncheckin.jl --treino")
        return false
    end
    
    try
        config = load_config(CONFIG_PATH)
        validate_config(config)
        
        println("\n🧠 Modelo:")
        println("   - Arquitetura: $(config["model"]["model_architecture"])")
        println("   - Tamanho da imagem: $(config["model"]["img_width"])x$(config["model"]["img_height"])")
        println("   - Número de classes: $(config["model"]["num_classes"])")
        println("   - Augmentação usada: $(config["model"]["augmentation_used"] ? "Sim" : "Não")")
        
        println("\n🎓 Treinamento:")
        println("   - Epochs treinados: $(config["training"]["epochs_trained"])")
        println("   - Learning rate: $(config["training"]["learning_rate"])")
        println("   - Batch size: $(config["training"]["batch_size"])")
        println("   - Acurácia final: $(round(config["training"]["final_accuracy"]*100, digits=2))%")
        println("   - Melhor epoch: $(config["training"]["best_epoch"])")
        
        if haskey(config, "training_stats")
            stats = config["training_stats"]
            println("   - Imagens de treino: $(stats["total_training_images"])")
            println("   - Imagens de validação: $(stats["total_validation_images"])")
            println("   - Early stopping: $(stats["early_stopped"] ? "Ativo" : "Completo")")
            if haskey(stats, "training_duration_minutes")
                println("   - Duração do treino: $(round(stats["training_duration_minutes"], digits=1)) min")
            end
        end
        
        println("\n📊 Dados:")
        println("   - Caminho dos dados: $(config["data"]["data_path"])")
        println("   - Pessoas reconhecidas: $(length(config["data"]["person_names"]))")
        for (i, name) in enumerate(config["data"]["person_names"])
            println("     $i. $name")
        end
        println("   - Criado em: $(config["data"]["timestamp"])")
        
        println("\n🔧 Metadados:")
        println("   - Versão: $(config["metadata"]["version"])")
        println("   - Criado por: $(config["metadata"]["created_by"])")
        if haskey(config["metadata"], "last_saved")
            println("   - Última atualização: $(config["metadata"]["last_saved"])")
        end
        
        return true
        
    catch e
        println("❌ Erro ao carregar informações: $e")
        return false
    end
end

# Função para exportar configuração (comando --export-config)
function export_config_command(output_path::String = "modelo_config_export.toml")
    println("📤 Exportando configuração do modelo...")
    
    if !isfile(CONFIG_PATH)
        println("❌ Configuração não encontrada. Execute primeiro: julia cnncheckin.jl --treino")
        return false
    end
    
    try
        # Carregar configuração atual
        config = load_config(CONFIG_PATH)
        
        # Adicionar informações de exportação
        config["export"] = Dict(
            "exported_at" => string(Dates.now()),
            "exported_from" => CONFIG_PATH,
            "export_version" => "1.0"
        )
        
        # Salvar cópia
        success = save_config(config, output_path)
        
        if success
            println("✅ Configuração exportada para: $output_path")
            println("📋 Você pode editar este arquivo e importá-lo depois")
        end
        
        return success
        
    catch e
        println("❌ Erro ao exportar configuração: $e")
        return false
    end
end

# Função para importar configuração (comando --import-config)
function import_config_command(import_path::String)
    println("📥 Importando configuração...")
    
    if !isfile(import_path)
        error("Arquivo de configuração não encontrado: $import_path")
    end
    
    try
        # Carregar configuração do arquivo especificado
        imported_config = TOML.parsefile(import_path)
        
        # Validar configuração importada
        validate_config(imported_config)
        
        # Atualizar timestamp
        imported_config["metadata"]["last_imported"] = string(Dates.now())
        imported_config["metadata"]["imported_from"] = import_path
        
        # Salvar como configuração ativa
        success = save_config(imported_config, CONFIG_PATH)
        
        if success
            println("✅ Configuração importada com sucesso!")
            println("📋 Nova configuração ativa salva em: $CONFIG_PATH")
            println("⚠️  Atenção: Certifique-se de que o modelo correspondente existe!")
        end
        
        return success
        
    catch e
        println("❌ Erro ao importar configuração: $e")
        return false
    end
end

# Função para validar modelo e configuração (comando --validate)
function validate_command()
    println("🔍 Validando modelo e configuração...")
    
    errors = []
    warnings = []
    
    # 1. Verificar se arquivos existem
    if !isfile(CONFIG_PATH)
        push!(errors, "Arquivo de configuração não encontrado: $CONFIG_PATH")
    end
    
    if !isfile(MODEL_PATH)
        push!(errors, "Arquivo do modelo não encontrado: $MODEL_PATH")
    end
    
    if !isempty(errors)
        println("❌ Erros encontrados:")
        for error in errors
            println("   - $error")
        end
        return false
    end
    
    try
        # 2. Validar configuração
        config = load_config(CONFIG_PATH)
        validate_config(config)
        println("✅ Configuração válida")
        
        # 3. Verificar consistência dos dados
        data_path = config["data"]["data_path"]
        if !isdir(data_path)
            push!(warnings, "Diretório de dados não encontrado: $data_path")
        else
            # Contar pessoas no diretório
            image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
            found_people = Set{String}()
            
            for filename in readdir(data_path)
                file_ext = lowercase(splitext(filename)[2])
                if file_ext in image_extensions
                    person_name = extract_person_name(filename)
                    push!(found_people, person_name)
                end
            end
            
            config_people = Set(config["data"]["person_names"])
            
            if found_people != config_people
                missing_in_config = setdiff(found_people, config_people)
                missing_in_data = setdiff(config_people, found_people)
                
                if !isempty(missing_in_config)
                    push!(warnings, "Pessoas no diretório não estão na configuração: $(join(missing_in_config, ", "))")
                end
                
                if !isempty(missing_in_data)
                    push!(warnings, "Pessoas na configuração não estão no diretório: $(join(missing_in_data, ", "))")
                end
            else
                println("✅ Dados consistentes com a configuração")
            end
        end
        
        # 4. Tentar carregar o modelo
        try
            model, person_names, _ = load_model_and_config(MODEL_PATH, CONFIG_PATH)
            println("✅ Modelo carregado com sucesso")
        catch e
            push!(errors, "Erro ao carregar modelo: $e")
        end
        
        # 5. Exibir resultados
        if !isempty(warnings)
            println("⚠️  Avisos encontrados:")
            for warning in warnings
                println("   - $warning")
            end
        end
        
        if !isempty(errors)
            println("❌ Erros encontrados:")
            for error in errors
                println("   - $error")
            end
            return false
        end
        
        if isempty(warnings) && isempty(errors)
            println("🎉 Modelo e configuração estão válidos e consistentes!")
        end
        
        return true
        
    catch e
        println("❌ Erro durante validação: $e")
        return false
    end
end

# Função para exibir ajuda
function show_help()
    println("🎯 Sistema de Reconhecimento Facial CNN Check-in v2.0")
    println()
    println("Uso:")
    println("  julia cnncheckin.jl --treino                         # Treinar o modelo")
    println("  julia cnncheckin.jl --identifica <imagem>            # Identificar pessoa")
    println("  julia cnncheckin.jl --info                           # Informações do modelo")
    println("  julia cnncheckin.jl --validate                       # Validar modelo/config")
    println("  julia cnncheckin.jl --export-config [arquivo]        # Exportar configuração")
    println("  julia cnncheckin.jl --import-config <arquivo>        # Importar configuração")
    println("  julia cnncheckin.jl --help                           # Mostrar esta ajuda")
    println()
    println("Exemplos:")
    println("  julia cnncheckin.jl --treino")
    println("  julia cnncheckin.jl --identifica ../dados/teste.jpg")
    println("  julia cnncheckin.jl --info")
    println("  julia cnncheckin.jl --export-config backup_config.toml")
    println("  julia cnncheckin.jl --import-config custom_config.toml")
    println()
    println("Estrutura de diretórios esperada:")
    println("  $DATA_PATH/")
    println("    ├── pessoa1-001.jpg")
    println("    ├── pessoa1-002.jpg")
    println("    ├── pessoa2-001.jpg")
    println("    └── ...")
    println()
    println("Formatos suportados: .jpg, .jpeg, .png, .bmp, .tiff")
    println()
    println("🔧 Configurações atuais:")
    println("  - Tamanho da imagem: $(IMG_SIZE)")
    println("  - Batch size: $BATCH_SIZE")
    println("  - Epochs máximos: $EPOCHS")
    println("  - Learning rate: $LEARNING_RATE")
    println("  - Arquivo do modelo: $MODEL_PATH")
    println("  - Arquivo de configuração: $CONFIG_PATH")
    println()
    println("📄 Arquivos gerados:")
    println("  - $MODEL_PATH    # Pesos neurais (JLD2)")
    println("  - $CONFIG_PATH   # Configurações (TOML)")
    println()
    println("💡 Novidades v2.0:")
    println("  - Configurações em formato TOML (editável)")
    println("  - Comando --info para detalhes do modelo")
    println("  - Comando --validate para verificar consistência")
    println("  - Comandos de export/import de configurações")
    println("  - Melhor rastreamento de métricas de treinamento")
end

# Função principal para parsing de argumentos - EXPANDIDA
function parse_args_and_run()
    if length(ARGS) == 0
        show_help()
        return
    end
    
    command = ARGS[1]
    
    if command == "--treino"
        println("🚀 Iniciando modo treinamento...")
        success = train_command()
        if success
            println("🎉 Treinamento concluído com sucesso!")
        else
            println("💥 Falha no treinamento")
        end
        
    elseif command == "--identifica"
        if length(ARGS) < 2
            println("❌ Erro: Caminho da imagem não fornecido")
            println("Uso: julia cnncheckin.jl --identifica <caminho_da_imagem>")
            return
        end
        
        image_path = ARGS[2]
        success = identify_command(image_path)
        if success
            println("✅ Identificação concluída!")
        else
            println("💥 Falha na identificação")
        end
        
    elseif command == "--info"
        success = info_command()
        if !success
            println("💥 Falha ao obter informações")
        end
        
    elseif command == "--validate"
        success = validate_command()
        if success
            println("✅ Validação concluída!")
        else
            println("💥 Falha na validação")
        end
        
    elseif command == "--export-config"
        output_path = length(ARGS) >= 2 ? ARGS[2] : "modelo_config_export.toml"
        success = export_config_command(output_path)
        if success
            println("✅ Exportação concluída!")
        else
            println("💥 Falha na exportação")
        end
        
    elseif command == "--import-config"
        if length(ARGS) < 2
            println("❌ Erro: Caminho do arquivo de configuração não fornecido")
            println("Uso: julia cnncheckin.jl --import-config <arquivo_config.toml>")
            return
        end
        
        import_path = ARGS[2]
        success = import_config_command(import_path)
        if success
            println("✅ Importação concluída!")
        else
            println("💥 Falha na importação")
        end
        
    elseif command == "--help" || command == "-h"
        show_help()
        
    else
        println("❌ Comando não reconhecido: $command")
        println("Use --help para ver os comandos disponíveis")
        show_help()
    end
end

# Função principal original (para compatibilidade)
function main()
    println("⚠️  Aviso: Use os novos comandos CLI para melhor experiência!")
    println("Execute: julia cnncheckin.jl --help")
    
    return train_command()
end

# Função de saudação original
greet() = print("Hello World!")

# Exportar funções principais
export main, predict_person, load_model_and_config, save_model_and_config, greet, train_command, identify_command, info_command

end # module cnncheckin

# Executar o parser de argumentos se for o arquivo principal
if abspath(PROGRAM_FILE) == @__FILE__
    cnncheckin.parse_args_and_run()
end

# Exemplos de uso:

# # Treinar modelo (gera .jld2 + .toml)
# julia cnncheckin.jl --treino

# julia cnncheckin.jl --identifica ../../../dados/fotos_teste/534770020_18019526477744454_2931624826193581596_n.jpg
# julia cnncheckin.jl --info
# # Validar consistência
# julia cnncheckin.jl --validate
# julia cnncheckin.jl --export-config backup.toml
# julia cnncheckin.jl --import-config custom.toml
 
 
# # Identificar pessoa
# julia cnncheckin.jl --identifica foto.jpg

# # Ver informações do modelo

# # Fazer backup da configuração
# julia cnncheckin.jl --export-config backup_2025.toml

# # Importar configuração personalizada
# julia cnncheckin.jl --import-config custom_config.toml