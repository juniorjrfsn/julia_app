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

# Função para extrair informações do modelo para TOML (ATUALIZADA)
function extract_model_info_for_toml(model, person_names::Vector{String})
    model_info = Dict{String, Any}()
    
    # Informações gerais do modelo
    model_info["model_summary"] = Dict(
        "total_layers" => length(model),
        "model_type" => "CNN",
        "input_shape" => collect(IMG_SIZE) .|> Int, # Convert tuple to array
        "output_classes" => length(person_names),
        "created_at" => string(Dates.now())
    )
    
    # Informações detalhadas das camadas
    layer_info = []
    for (i, layer) in enumerate(model)
        layer_dict = Dict{String, Any}(
            "layer_number" => i,
            "layer_type" => string(typeof(layer)),
            "trainable" => true
        )
        
        # Extrair informações específicas por tipo de camada
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
    
    # Resumo dos pesos
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
    
    # Mapeamento pessoa -> ID
    person_mappings = Dict{String, Int}()
    for (i, name) in enumerate(person_names)
        person_mappings[name] = i
    end
    model_info["person_mappings"] = person_mappings
    
    # Placeholder para exemplos de predição
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

# Função para adicionar exemplo de predição ao TOML
function add_prediction_example_to_toml(image_path::String, predicted_person::String, 
                                       confidence::Float64, actual_person::String = "")
    model_data = load_model_data_toml(MODEL_DATA_TOML_PATH)
    if model_data === nothing
        return false
    end
    
    example = Dict(
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
        model_data["prediction_examples"] = []
    end
    
    push!(model_data["prediction_examples"], example)
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
            "description" => "Configurações do modelo de reconhecimento facial"
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

# Função para carregar dados das imagens
function load_face_data(data_path::String; use_augmentation::Bool = true)
    println("📄 Carregando dados das imagens...")
    
    if !isdir(data_path)
        error("Diretório $data_path não encontrado!")
    end
    
    person_images = Dict{String, Vector{Array{Float32, 3}}}()
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    
    for filename in readdir(data_path)
        file_ext = lowercase(splitext(filename)[2])
        if file_ext in image_extensions
            img_path = joinpath(data_path, filename)
            if !validate_image_file(img_path)
                continue
            end
            person_name = extract_person_name(filename)
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
                println("⚠️ Falha ao carregar: $filename")
            end
        end
    end
    
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

# Função para criar datasets balanceados
function create_datasets(people_data::Vector{PersonData}, split_ratio::Float64 = 0.8)
    println("📊 Criando datasets de treino e validação...")
    
    train_images = Vector{Array{Float32, 3}}()
    train_labels = Vector{Int}()
    val_images = Vector{Array{Float32, 3}}()
    val_labels = Vector{Int}()
    
    for person in people_data
        n_imgs = length(person.images)
        n_train = max(1, Int(floor(n_imgs * split_ratio)))
        indices = randperm(n_imgs)
        
        for i in 1:n_train
            push!(train_images, person.images[indices[i]])
            push!(train_labels, person.label)
        end
        
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

# Função para criar batches (ATUALIZADA)
function create_batches(images, labels, batch_size)
    batches = []
    n_samples = length(images)
    if n_samples == 0
        return batches
    end
    max_label = maximum(labels)
    min_label = minimum(labels)
    label_range = min_label:max_label
    
    for i in 1:batch_size:n_samples
        end_idx = min(i + batch_size - 1, n_samples)
        batch_images = images[i:end_idx]
        batch_labels = labels[i:end_idx]
        batch_tensor = cat(batch_images..., dims=4)
        try
            batch_labels_onehot = Flux.onehotbatch(batch_labels, label_range)
            push!(batches, (batch_tensor, batch_labels_onehot))
        catch e
            println("❌ Erro ao criar one-hot encoding para batch $i-$end_idx: $e")
            continue
        end
    end
    
    return batches
end

# Arquitetura da CNN
function create_cnn_model(num_classes::Int, input_size::Tuple{Int, Int} = IMG_SIZE)
    final_size = div(div(div(div(input_size[1], 2), 2), 2), 2)
    final_features = 128 * final_size * final_size
    
    return Chain(
        Conv((3, 3), 3 => 32, relu, pad=1),
        BatchNorm(32),
        Dropout(0.1),
        MaxPool((2, 2)),
        Conv((3, 3), 32 => 64, relu, pad=1),
        BatchNorm(64),
        Dropout(0.1),
        MaxPool((2, 2)),
        Conv((3, 3), 64 => 128, relu, pad=1),
        BatchNorm(128),
        Dropout(0.2),
        MaxPool((2, 2)),
        Conv((3, 3), 128 => 128, relu, pad=1),
        BatchNorm(128),
        Dropout(0.2),
        MaxPool((2, 2)),
        Flux.flatten,
        Dense(final_features, 256, relu),
        Dropout(0.5),
        Dense(256, 64, relu),
        Dropout(0.4),
        Dense(64, num_classes)
    )
end

# Função para calcular acurácia
function accuracy(model, data_loader)
    correct = 0
    total = 0
    for (x, y) in data_loader
        try
            ŷ = softmax(model(x))
            pred = Flux.onecold(ŷ)
            true_labels = Flux.onecold(y)
            correct += sum(pred .== true_labels)
            total += length(true_labels)
        catch e
            println("⚠️ Erro ao calcular acurácia para batch: $e")
            continue
        end
    end
    return total > 0 ? correct / total : 0.0
end

# Função de treinamento
function train_model(model, train_data, val_data, epochs, learning_rate)
    println("🚀 Iniciando treinamento...")
    
    optimizer = ADAM(learning_rate, (0.9, 0.999), 1e-8)
    opt_state = Flux.setup(optimizer, model)
    
    train_losses = Float64[]
    val_accuracies = Float64[]
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    patience_limit = 10
    
    for epoch in 1:epochs
        epoch_loss = 0.0
        num_batches = 0
        
        for (x, y) in train_data
            try
                loss, grads = Flux.withgradient(model) do m
                    ŷ = m(x)
                    Flux.logitcrossentropy(ŷ, y)
                end
                Flux.update!(opt_state, model, grads[1])
                epoch_loss += loss
                num_batches += 1
            catch e
                println("⚠️ Erro no treinamento do batch na epoch $epoch: $e")
                continue
            end
        end
        
        avg_loss = num_batches > 0 ? epoch_loss / num_batches : 0.0
        val_acc = accuracy(model, val_data)
        
        push!(train_losses, avg_loss)
        push!(val_accuracies, val_acc)
        
        if val_acc > best_val_acc
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
        else
            patience_counter += 1
        end
        
        if epoch % 5 == 0 || epoch == 1
            println("Epoch $epoch/$epochs - Loss: $(round(avg_loss, digits=4)) - Val Acc: $(round(val_acc*100, digits=2))% - Best: $(round(best_val_acc*100, digits=2))% (Epoch $best_epoch)")
        end
        
        if patience_counter >= patience_limit
            println("🛑 Early stopping - sem melhoria por $patience_limit epochs")
            break
        end
        
        if val_acc >= 0.85 && val_acc < 0.98
            println("🎯 Acurácia boa alcançada sem overfitting!")
            break
        end
    end
    
    return train_losses, val_accuracies, best_val_acc, best_epoch
end

# Função para salvar modelo, configuração e dados do modelo
function save_model_and_config(model, person_names, model_filepath::String, config_filepath::String, training_info::Dict)
    println("💾 Salvando modelo, configuração e dados do modelo...")
    
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
    
    model_data_saved = save_model_data_toml(model, person_names, MODEL_DATA_TOML_PATH)
    if !model_data_saved
        println("⚠️ Aviso: Falha ao salvar dados do modelo em TOML")
    end
    
    config = create_default_config()
    config["model"]["num_classes"] = length(person_names)
    config["model"]["augmentation_used"] = training_info["augmentation_used"]
    config["training"]["epochs_trained"] = training_info["epochs_trained"]
    config["training"]["final_accuracy"] = training_info["final_accuracy"]
    config["training"]["best_epoch"] = training_info["best_epoch"]
    config["data"]["person_names"] = person_names
    config["data"]["timestamp"] = string(Dates.now())
    config["training_stats"] = Dict(
        "total_training_images" => training_info["total_training_images"],
        "total_validation_images" => training_info["total_validation_images"],
        "early_stopped" => training_info["early_stopped"],
        "training_duration_minutes" => get(training_info, "duration_minutes", 0.0)
    )
    config["files"] = Dict(
        "model_weights" => MODEL_PATH,
        "model_data_toml" => MODEL_DATA_TOML_PATH,
        "config_toml" => CONFIG_PATH
    )
    
    config_saved = save_config(config, config_filepath)
    return config_saved && model_data_saved
end

# Função para carregar modelo, configuração e dados do modelo
function load_model_and_config(model_filepath::String, config_filepath::String)
    println("📂 Carregando modelo, configuração e dados do modelo...")
    
    config = load_config(config_filepath)
    validate_config(config)
    
    if !isfile(model_filepath)
        error("Arquivo do modelo não encontrado: $model_filepath")
    end
    
    try
        data = load(model_filepath)
        model_data = data["model_data"]
        model_state = model_data["model_state"]
        person_names = config["data"]["person_names"]
        num_classes = config["model"]["num_classes"]
        model_data_toml = load_model_data_toml(MODEL_DATA_TOML_PATH)
        
        println("✅ Modelo e configuração carregados com sucesso!")
        println("📊 Informações do modelo:")
        println("   - Classes: $num_classes")
        println("   - Pessoas: $(join(person_names, ", "))")
        println("   - Acurácia: $(round(config["training"]["final_accuracy"]*100, digits=2))%")
        println("   - Criado em: $(config["data"]["timestamp"])")
        
        if model_data_toml !== nothing
            println("   - Dados TOML disponíveis: ✅")
            total_params = get(get(model_data_toml, "weights_summary", Dict()), "total_parameters", 0)
            if total_params > 0
                println("   - Total de parâmetros: $(total_params)")
            end
        else
            println("   - Dados TOML disponíveis: ❌")
        end
        
        return model_state, person_names, config, model_data_toml
    catch e
        error("Erro ao carregar modelo: $e")
    end
end

# Função para fazer predição e registrar exemplo (ATUALIZADA)
function predict_person(model, person_names, img_path::String; save_example::Bool = true)
    img_arrays = preprocess_image(img_path; augment=false)
    
    if img_arrays === nothing || length(img_arrays) == 0
        println("❌ Não foi possível processar a imagem")
        return nothing, 0.0
    end
    
    img_array = img_arrays[1]
    img_tensor = reshape(img_array, size(img_array)..., 1)
    
    try
        logits = model(img_tensor)
        if size(logits, 1) != length(person_names)
            error("Dimensão de saída do modelo ($size(logits, 1)) não corresponde ao número de classes ($(length(person_names)))")
        end
        prediction = softmax(logits)
        pred_probs = vec(prediction)
        pred_class = argmax(pred_probs) # 1-based index
        confidence = pred_probs[pred_class]
        person_name = pred_class <= length(person_names) ? person_names[pred_class] : "Desconhecido"
        
        if save_example
            add_prediction_example_to_toml(img_path, person_name, Float64(confidence))
        end
        
        return person_name, confidence
    catch e
        println("❌ Erro ao realizar predição: $e")
        return nothing, 0.0
    end
end

# Função de treinamento (comando --treino)
function train_command()
    println("🎯 Sistema de Reconhecimento Facial - Modo Treinamento")
    
    start_time = time()
    
    try
        people_data, person_names = load_face_data(DATA_PATH; use_augmentation=true)
        if length(people_data) == 0
            error("Nenhum dado encontrado!")
        end
        
        num_classes = length(person_names)
        println("👥 Total de pessoas: $num_classes")
        total_images = sum(length(person.images) for person in people_data)
        println("📊 Total de imagens (com augmentation): $total_images")
        
        (train_images, train_labels), (val_images, val_labels) = create_datasets(people_data)
        train_batches = create_batches(train_images, train_labels, BATCH_SIZE)
        val_batches = create_batches(val_images, val_labels, BATCH_SIZE)
        
        if length(train_batches) == 0
            error("Não foi possível criar batches de treinamento!")
        end
        
        println("🧠 Criando modelo CNN...")
        model = create_cnn_model(num_classes)
        train_losses, val_accuracies, best_val_acc, best_epoch = train_model(model, train_batches, val_batches, EPOCHS, LEARNING_RATE)
        
        end_time = time()
        duration_minutes = (end_time - start_time) / 60
        
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
        
        println("🎉 Treinamento concluído!")
        println("📊 Resultados:")
        println("   - Melhor acurácia: $(round(best_val_acc*100, digits=2))% (Epoch $best_epoch)")
        println("   - Epochs treinados: $(training_info["epochs_trained"])/$EPOCHS")
        println("   - Duração: $(round(duration_minutes, digits=1)) minutos")
        println("   - Early stopping: $(training_info["early_stopped"] ? "Sim" : "Não")")
        
        success = save_model_and_config(model, person_names, MODEL_PATH, CONFIG_PATH, training_info)
        
        if success
            println("✅ Sistema treinado e salvo com sucesso!")
            println("📄 Arquivos gerados:")
            println("   - Configuração: $CONFIG_PATH")
            println("   - Pesos do modelo: $MODEL_PATH")
            println("   - Dados do modelo: $MODEL_DATA_TOML_PATH")
        else
            println("⚠️ Modelo treinado mas houve erro ao salvar alguns arquivos")
        end
        return success
    catch e
        println("❌ Erro durante treinamento: $e")
        return false
    end
end

# Função de identificação
function identify_command(image_path::String)
    println("🔍 Sistema de Reconhecimento Facial - Modo Identificação")
    println("📸 Analisando imagem: $image_path")
    
    try
        if !isfile(image_path)
            error("Arquivo de imagem não encontrado: $image_path")
        end
        if !isfile(MODEL_PATH)
            error("Modelo não encontrado! Execute primeiro: julia cnncheckin.jl --treino")
        end
        if !isfile(CONFIG_PATH)
            error("Configuração não encontrada! Execute primeiro: julia cnncheckin.jl --treino")
        end
        
        model, person_names, config, model_data_toml = load_model_and_config(MODEL_PATH, CONFIG_PATH)
        person_name, confidence = predict_person(model, person_names, image_path; save_example=true)
        
        if person_name === nothing
            println("❌ Não foi possível processar a imagem")
            return false
        end
        
        println("🎯 Resultado da Identificação:")
        println("   👤 Pessoa: $person_name")
        println("   📈 Confiança: $(round(confidence*100, digits=2))%")
        
        println("\n📊 Informações do Modelo:")
        println("   🧠 Acurácia do modelo: $(round(config["training"]["final_accuracy"]*100, digits=2))%")
        println("   📅 Treinado em: $(config["data"]["timestamp"])")
        println("   🎓 Melhor epoch: $(config["training"]["best_epoch"])")
        
        if model_data_toml !== nothing
            total_params = get(get(model_data_toml, "weights_summary", Dict()), "total_parameters", 0)
            if total_params > 0
                model_size = get(get(model_data_toml, "weights_summary", Dict()), "model_size_mb", 0.0)
                println("   🔢 Parâmetros do modelo: $(total_params)")
                println("   💾 Tamanho estimado: $(model_size) MB")
            end
            examples = get(model_data_toml, "prediction_examples", [])
            if length(examples) > 0
                println("\n📝 Exemplo salvo como predição #$(length(examples))")
            end
        end
        
        if confidence >= 0.7
            println("✅ Identificação com alta confiança")
        elseif confidence >= 0.5
            println("⚠️ Identificação com confiança média")
        else
            println("❓ Identificação com baixa confiança - pode ser pessoa desconhecida")
        end
        return true
    catch e
        println("❌ Erro durante identificação: $e")
        return false
    end
end

# Função para exibir informações
function info_command()
    println("📋 Informações do Modelo de Reconhecimento Facial")
    
    if !isfile(CONFIG_PATH)
        println("❌ Configuração não encontrada. Execute primeiro: julia cnncheckin.jl --treino")
        return false
    end
    
    try
        config = load_config(CONFIG_PATH)
        validate_config(config)
        model_data_toml = load_model_data_toml(MODEL_DATA_TOML_PATH)
        
        println("\n🧠 Modelo:")
        println("   - Arquitetura: $(config["model"]["model_architecture"])")
        println("   - Tamanho da imagem: $(config["model"]["img_width"])x$(config["model"]["img_height"])")
        println("   - Número de classes: $(config["model"]["num_classes"])")
        println("   - Augmentação usada: $(config["model"]["augmentation_used"] ? "Sim" : "Não")")
        
        if model_data_toml !== nothing
            model_summary = get(model_data_toml, "model_summary", Dict())
            weights_summary = get(model_data_toml, "weights_summary", Dict())
            if !isempty(model_summary)
                println("   - Total de camadas: $(get(model_summary, "total_layers", "N/A"))")
                println("   - Formato entrada: $(get(model_summary, "input_shape", "N/A"))")
            end
            if !isempty(weights_summary)
                total_params = get(weights_summary, "total_parameters", 0)
                model_size = get(weights_summary, "model_size_mb", 0.0)
                if total_params > 0
                    println("   - Parâmetros totais: $(total_params)")
                    println("   - Tamanho estimado: $(model_size) MB")
                end
            end
        end
        
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
        
        if model_data_toml !== nothing
            person_mappings = get(model_data_toml, "person_mappings", Dict())
            if !isempty(person_mappings)
                println("\n🔗 Mapeamentos:")
                for (name, id) in person_mappings
                    println("   - $name → ID $id")
                end
            end
            examples = get(model_data_toml, "prediction_examples", [])
            if !isempty(examples)
                println("\n📝 Últimas Predições ($(length(examples))):")
                recent_examples = examples[max(1, length(examples)-4):end]
                for (i, example) in enumerate(recent_examples)
                    timestamp = get(example, "timestamp", "N/A")
                    predicted = get(example, "predicted_person", "N/A")
                    confidence = get(example, "confidence", 0.0)
                    filename = get(example, "image_filename", "N/A")
                    println("   $(length(examples)-length(recent_examples)+i). $filename")
                    println("      → $predicted ($(round(confidence*100, digits=1))%)")
                    println("      📅 $timestamp")
                end
                if length(examples) > 5
                    println("   ... e mais $(length(examples)-5) exemplos")
                end
            end
        end
        
        println("\n📧 Metadados:")
        println("   - Versão: $(config["metadata"]["version"])")
        println("   - Criado por: $(config["metadata"]["created_by"])")
        if haskey(config["metadata"], "last_saved")
            println("   - Última atualização: $(config["metadata"]["last_saved"])")
        end
        
        if haskey(config, "files")
            files = config["files"]
            println("\n📁 Arquivos:")
            for (key, filepath) in files
                status = isfile(filepath) ? "✅" : "❌"
                println("   - $(key): $(filepath) $status")
            end
        else
            println("\n📁 Arquivos:")
            println("   - Configuração: $CONFIG_PATH $(isfile(CONFIG_PATH) ? "✅" : "❌")")
            println("   - Modelo: $MODEL_PATH $(isfile(MODEL_PATH) ? "✅" : "❌")")
            println("   - Dados TOML: $MODEL_DATA_TOML_PATH $(isfile(MODEL_DATA_TOML_PATH) ? "✅" : "❌")")
        end
        return true
    catch e
        println("❌ Erro ao carregar informações: $e")
        return false
    end
end

# Função para exibir detalhes dos dados do modelo
function model_data_command()
    println("🔬 Dados Detalhados do Modelo")
    
    model_data_toml = load_model_data_toml(MODEL_DATA_TOML_PATH)
    if model_data_toml === nothing
        println("❌ Dados do modelo não encontrados. Execute primeiro: julia cnncheckin.jl --treino")
        return false
    end
    
    try
        model_summary = get(model_data_toml, "model_summary", Dict())
        if !isempty(model_summary)
            println("\n📊 Resumo do Modelo:")
            for (key, value) in model_summary
                println("   - $(key): $(value)")
            end
        end
        
        layer_info = get(model_data_toml, "layer_info", [])
        if !isempty(layer_info)
            println("\n🏗️ Arquitetura das Camadas:")
            for layer in layer_info
                layer_num = get(layer, "layer_number", "?")
                layer_type = get(layer, "layer_type", "Unknown")
                println("   Camada $layer_num: $layer_type")
                if haskey(layer, "kernel_size")
                    println("      - Kernel: $(layer["kernel_size"])")
                    println("      - Canais: $(layer["input_channels"]) → $(layer["output_channels"])")
                elseif haskey(layer, "input_size")
                    println("      - Entrada: $(layer["input_size"])")
                    println("      - Saída: $(layer["output_size"])")
                elseif haskey(layer, "pool_size")
                    println("      - Pool size: $(layer["pool_size"])")
                elseif haskey(layer, "dropout_rate")
                    println("      - Taxa dropout: $(layer["dropout_rate"])")
                elseif haskey(layer, "num_features")
                    println("      - Features: $(layer["num_features"])")
                end
            end
        end
        
        weights_summary = get(model_data_toml, "weights_summary", Dict())
        if !isempty(weights_summary)
            println("\n⚖️ Estatísticas dos Pesos:")
            total_params = get(weights_summary, "total_parameters", 0)
            model_size = get(weights_summary, "model_size_mb", 0.0)
            println("   - Parâmetros totais: $(total_params)")
            println("   - Tamanho estimado: $(model_size) MB")
            layer_stats = get(weights_summary, "layer_statistics", Dict())
            if !isempty(layer_stats)
                println("\n   Estatísticas por camada:")
                for (layer_key, stats) in layer_stats
                    println("   - $layer_key:")
                    println("     Shape: $(get(stats, "shape", "N/A"))")
                    println("     Count: $(get(stats, "count", "N/A"))")
                    println("     Mean: $(round(get(stats, "mean", 0.0), digits=6))")
                    println("     Std: $(round(get(stats, "std", 0.0), digits=6))")
                end
            end
        end
        
        person_mappings = get(model_data_toml, "person_mappings", Dict())
        if !isempty(person_mappings)
            println("\n👥 Mapeamento de Pessoas:")
            sorted_mappings = sort(collect(person_mappings), by=x->x[2])
            for (name, id) in sorted_mappings
                println("   ID $id: $name")
            end
        end
        
        examples = get(model_data_toml, "prediction_examples", [])
        if !isempty(examples)
            println("\n📝 Histórico de Predições ($(length(examples)) total):")
            confidences = [get(ex, "confidence", 0.0) for ex in examples]
            if !isempty(confidences)
                avg_conf = mean(confidences)
                min_conf = minimum(confidences)
                max_conf = maximum(confidences)
                println("   📊 Estatísticas de confiança:")
                println("     - Média: $(round(avg_conf*100, digits=1))%")
                println("     - Mínima: $(round(min_conf*100, digits=1))%")
                println("     - Máxima: $(round(max_conf*100, digits=1))%")
            end
            println("\n   🕐 Últimas predições:")
            recent_examples = examples[max(1, length(examples)-9):end]
            for (i, example) in enumerate(reverse(recent_examples))
                timestamp = get(example, "timestamp", "N/A")
                predicted = get(example, "predicted_person", "N/A")
                confidence = get(example, "confidence", 0.0)
                filename = get(example, "image_filename", "N/A")
                println("   $(i). $filename")
                println("      → $predicted ($(round(confidence*100, digits=1))%)")
                println("      📅 $(timestamp[1:19])")
            end
        end
        
        metadata = get(model_data_toml, "metadata", Dict())
        if !isempty(metadata)
            println("\n📋 Metadados dos Dados do Modelo:")
            for (key, value) in metadata
                println("   - $(key): $(value)")
            end
        end
        return true
    catch e
        println("❌ Erro ao exibir dados do modelo: $e")
        return false
    end
end

# Função para exportar configuração
function export_config_command(output_path::String = "modelo_config_export.toml")
    println("📤 Exportando configuração do modelo...")
    
    if !isfile(CONFIG_PATH)
        println("❌ Configuração não encontrada. Execute primeiro: julia cnncheckin.jl --treino")
        return false
    end
    
    try
        config = load_config(CONFIG_PATH)
        config["export"] = Dict(
            "exported_at" => string(Dates.now()),
            "exported_from" => CONFIG_PATH,
            "export_version" => "1.0"
        )
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

# Função para exportar dados do modelo
function export_model_data_command(output_path::String = "modelo_data_export.toml")
    println("📤 Exportando dados do modelo...")
    
    model_data_toml = load_model_data_toml(MODEL_DATA_TOML_PATH)
    if model_data_toml === nothing
        println("❌ Dados do modelo não encontrados. Execute primeiro: julia cnncheckin.jl --treino")
        return false
    end
    
    try
        model_data_toml["export"] = Dict(
            "exported_at" => string(Dates.now()),
            "exported_from" => MODEL_DATA_TOML_PATH,
            "export_version" => "1.0"
        )
        open(output_path, "w") do io
            TOML.print(io, model_data_toml)
        end
        println("✅ Dados do modelo exportados para: $output_path")
        println("📋 Este arquivo contém informações técnicas detalhadas do modelo")
        return true
    catch e
        println("❌ Erro ao exportar dados do modelo: $e")
        return false
    end
end

# Função para importar configuração
function import_config_command(import_path::String)
    println("📥 Importando configuração...")
    
    if !isfile(import_path)
        error("Arquivo de configuração não encontrado: $import_path")
    end
    
    try
        imported_config = TOML.parsefile(import_path)
        validate_config(imported_config)
        imported_config["metadata"]["last_imported"] = string(Dates.now())
        imported_config["metadata"]["imported_from"] = import_path
        success = save_config(imported_config, CONFIG_PATH)
        if success
            println("✅ Configuração importada com sucesso!")
            println("📋 Nova configuração ativa salva em: $CONFIG_PATH")
            println("⚠️ Atenção: Certifique-se de que o modelo correspondente existe!")
        end
        return success
    catch e
        println("❌ Erro ao importar configuração: $e")
        return false
    end
end

# Função para validar modelo, configuração e dados
function validate_command()
    println("🔍 Validando modelo, configuração e dados...")
    
    errors = []
    warnings = []
    
    if !isfile(CONFIG_PATH)
        push!(errors, "Arquivo de configuração não encontrado: $CONFIG_PATH")
    end
    if !isfile(MODEL_PATH)
        push!(errors, "Arquivo do modelo não encontrado: $MODEL_PATH")
    end
    if !isfile(MODEL_DATA_TOML_PATH)
        push!(warnings, "Arquivo de dados do modelo não encontrado: $MODEL_DATA_TOML_PATH")
    end
    
    if !isempty(errors)
        println("❌ Erros encontrados:")
        for error in errors
            println("   - $error")
        end
        return false
    end
    
    try
        config = load_config(CONFIG_PATH)
        validate_config(config)
        println("✅ Configuração válida")
        
        model_data_toml = load_model_data_toml(MODEL_DATA_TOML_PATH)
        if model_data_toml !== nothing
            println("✅ Dados do modelo TOML válidos")
            config_people = Set(config["data"]["person_names"])
            toml_people = Set(keys(get(model_data_toml, "person_mappings", Dict())))
            if config_people != toml_people
                push!(warnings, "Inconsistência entre pessoas na configuração e nos dados TOML")
            else
                println("✅ Consistência entre configuração e dados TOML")
            end
        else
            push!(warnings, "Não foi possível validar dados do modelo TOML")
        end
        
        data_path = config["data"]["data_path"]
        if !isdir(data_path)
            push!(warnings, "Diretório de dados não encontrado: $data_path")
        else
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
        
        try
            model, person_names, _, model_data_toml = load_model_and_config(MODEL_PATH, CONFIG_PATH)
            println("✅ Modelo carregado com sucesso")
        catch e
            push!(errors, "Erro ao carregar modelo: $e")
        end
        
        if !isempty(warnings)
            println("⚠️ Avisos encontrados:")
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
            println("🎉 Modelo, configuração e dados estão válidos e consistentes!")
        end
        return true
    catch e
        println("❌ Erro durante validação: $e")
        return false
    end
end

# Função para exibir ajuda
function show_help()
    println("🎯 Sistema de Reconhecimento Facial CNN Check-in v2.1")
    println()
    println("Uso:")
    println("  julia cnncheckin.jl --treino                         # Treinar o modelo")
    println("  julia cnncheckin.jl --identifica <imagem>            # Identificar pessoa")
    println("  julia cnncheckin.jl --info                           # Informações do modelo")
    println("  julia cnncheckin.jl --model-data                     # Dados técnicos do modelo")
    println("  julia cnncheckin.jl --validate                       # Validar modelo/config/dados")
    println("  julia cnncheckin.jl --export-config [arquivo]        # Exportar configuração")
    println("  julia cnncheckin.jl --export-model-data [arquivo]    # Exportar dados do modelo")
    println("  julia cnncheckin.jl --import-config <arquivo>        # Importar configuração")
    println("  julia cnncheckin.jl --help                           # Mostrar esta ajuda")
    println()
    println("Exemplos:")
    println("  julia cnncheckin.jl --treino")
    println("  julia cnncheckin.jl --identifica ../dados/teste.jpg")
    println("  julia cnncheckin.jl --info")
    println("  julia cnncheckin.jl --model-data")
    println("  julia cnncheckin.jl --export-config backup_config.toml")
    println("  julia cnncheckin.jl --export-model-data detalhes_modelo.toml")
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
    println()
    println("📄 Arquivos gerados:")
    println("  - $MODEL_PATH              # Pesos neurais (JLD2)")
    println("  - $CONFIG_PATH             # Configurações (TOML)")
    println("  - $MODEL_DATA_TOML_PATH    # Dados do modelo (TOML)")
    println()
    println("💡 Novidades v2.1:")
    println("  - Dados do modelo salvos em TOML editável")
    println("  - Comando --model-data para detalhes técnicos")
    println("  - Comando --export-model-data para backup de dados")
    println("  - Histórico de predições salvo automaticamente")
    println("  - Estatísticas detalhadas dos pesos e camadas")
    println("  - Validação expandida incluindo dados TOML")
    println("  - Mapeamento pessoa → ID em formato legível")
    println()
    println("📊 O que contém o arquivo TOML de dados do modelo:")
    println("  - [model_summary]: arquitetura, dimensões, tipos")
    println("  - [layer_info]: detalhes de cada camada (Conv, Dense, etc.)")
    println("  - [weights_summary]: estatísticas dos pesos, tamanho do modelo")
    println("  - [person_mappings]: nome da pessoa → ID numérico")
    println("  - [prediction_examples]: histórico das últimas 50 predições")
    println("  - [metadata]: timestamps, versões, arquivos relacionados")
end

# Função para parsing de argumentos
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
            println("📊 Execute 'julia cnncheckin.jl --info' para ver detalhes")
            println("🔬 Execute 'julia cnncheckin.jl --model-data' para dados técnicos")
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
    elseif command == "--model-data"
        success = model_data_command()
        if success
            println("✅ Dados técnicos exibidos!")
        else
            println("💥 Falha ao obter dados do modelo")
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
            println("✅ Exportação de configuração concluída!")
        else
            println("💥 Falha na exportação")
        end
    elseif command == "--export-model-data"
        output_path = length(ARGS) >= 2 ? ARGS[2] : "modelo_data_export.toml"
        success = export_model_data_command(output_path)
        if success
            println("✅ Exportação de dados do modelo concluída!")
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

# Função principal original
function main()
    println("⚠️ Aviso: Use os novos comandos CLI para melhor experiência!")
    println("Execute: julia cnncheckin.jl --help")
    return train_command()
end

# Função de saudação
greet() = print("Hello World!")

# Exportar funções principais
export main, predict_person, load_model_and_config, save_model_and_config, greet, train_command, identify_command, info_command, model_data_command

end # module cnncheckin

# Executar o parser de argumentos
if abspath(PROGRAM_FILE) == @__FILE__
    cnncheckin.parse_args_and_run()
end






# Exemplos de uso expandidos:

# # 1. TREINAR MODELO (gera 3 arquivos: .jld2 + 2 arquivos .toml)
# julia cnncheckin.jl --treino

# # 2. IDENTIFICAR PESSOA (salva exemplo no histórico)
# julia cnncheckin.jl --identifica ../../../dados/fotos_teste/pessoa1.jpg


# julia cnncheckin.jl --identifica ../../../dados/fotos_teste/ela.jpg

# # 3. VER INFORMAÇÕES GERAIS
# julia cnncheckin.jl --info

# # 4. VER DADOS TÉCNICOS DETALHADOS (NOVO)
# julia cnncheckin.jl --model-data

# # 5. VALIDAR TUDO (modelo + config + dados TOML)
# julia cnncheckin.jl --validate

# # 6. EXPORTAR CONFIGURAÇÃO
# julia cnncheckin.jl --export-config backup_config_2025.toml

# # 7. EXPORTAR DADOS DO MODELO (NOVO)
# julia cnncheckin.jl --export-model-data detalhes_modelo_2025.toml

# # 8. IMPORTAR CONFIGURAÇÃO PERSONALIZADA
# julia cnncheckin.jl --import-config custom_config.toml

# # 9. AJUDA EXPANDIDA
# julia cnncheckin.jl --help

# ARQUIVOS GERADOS:
# ├── face_recognition_model.jld2          # Pesos neurais (binário)
# ├── face_recognition_config.toml         # Configurações (editável)
# └── face_recognition_model_data.toml     # Dados técnicos (editável)

# CONTEÚDO DO ARQUIVO face_recognition_model_data.toml:
# - [model_summary]: arquitetura, dimensões, tipos
# - [layer_info]: detalhes de cada camada (Conv, Dense, etc.)
# - [weights_summary]: estatísticas dos pesos, tamanho do modelo
# - [person_mappings]: nome da pessoa → ID numérico
# - [prediction_examples]: histórico das últimas 50 predições
# - [metadata]: timestamps, versões, arquivos relacionados

# VANTAGENS:
# ✅ Dados do modelo totalmente editáveis e inspecionáveis
# ✅ Histórico automático de predições
# ✅ Estatísticas detalhadas para debugging
# ✅ Mapeamentos claros pessoa ↔ ID
# ✅ Validação expandida de consistência
# ✅ Backup e restauração de dados técnicos
# ✅ Compatibilidade total com versão anterior