# projeto: webcamcnn
# file: webcamcnn/src/pretrain_modified.jl

using Flux
using Statistics
using Random
using JLD2
using Dates

include("core.jl")
include("weights_manager.jl")
using .CNNCheckinCore

# Constante para o arquivo de pesos
const WEIGHTS_TOML_PATH = "model_weights.toml"

# Function to load initial training data (mantida igual)
function load_pretrain_data(data_path::String; use_augmentation::Bool = true)
    println("📂 Carregando dados de pré-treino...")
    
    if !isdir(data_path)
        error("Diretório $data_path não encontrado!")
    end
    
    person_images = Dict{String, Vector{Array{Float32, 3}}}()
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    
    for filename in readdir(data_path)
        file_ext = lowercase(splitext(filename)[2])
        if file_ext in image_extensions
            img_path = joinpath(data_path, filename)
            if !CNNCheckinCore.validate_image_file(img_path)
                continue
            end
            
            person_name = CNNCheckinCore.extract_person_name(filename)
            img_arrays = CNNCheckinCore.preprocess_image(img_path; augment=use_augmentation)
            
            if img_arrays !== nothing
                if !haskey(person_images, person_name)
                    person_images[person_name] = Vector{Array{Float32, 3}}()
                end
                for img_array in img_arrays
                    push!(person_images[person_name], img_array)
                end
                total_imgs = use_augmentation ? length(img_arrays) : 1
                println("✅ Carregado: $filename -> $person_name ($total_imgs variações)")
            else
                println("❌ Falha ao carregar: $filename")
            end
        end
    end
    
    people_data = Vector{CNNCheckinCore.PersonData}()
    person_names = sort(collect(keys(person_images)))
    
    # Create person data with proper indexing
    for (idx, person_name) in enumerate(person_names)
        images = person_images[person_name]
        if length(images) > 0
            push!(people_data, CNNCheckinCore.PersonData(person_name, images, idx, false))
            println("👤 Pessoa: $person_name - $(length(images)) imagens (Label: $idx)")
        end
    end
    
    return people_data, person_names
end

# Create balanced datasets for pre-training (mantida igual)
function create_pretrain_datasets(people_data::Vector{CNNCheckinCore.PersonData}, split_ratio::Float64 = 0.8)
    println("🔄 Criando datasets de treino e validação...")
    
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
        
        println("   - $(person.name): $n_train treino, $(n_imgs - n_train) validação")
    end
    
    println("📊 Dataset criado:")
    println("   - Treino: $(length(train_images)) imagens")
    println("   - Validação: $(length(val_images)) imagens")
    
    return (train_images, train_labels), (val_images, val_labels)
end

# Create training batches (mantida igual)
function create_pretrain_batches(images, labels, batch_size)
    batches = []
    n_samples = length(images)
    if n_samples == 0
        return batches
    end
    
    unique_labels = unique(labels)
    min_label = minimum(unique_labels)
    max_label = maximum(unique_labels)
    label_range = min_label:max_label
    
    println("🏷️  Labels únicos: $unique_labels")
    println("🏷️  Range de labels: $label_range")
    
    for i in 1:batch_size:n_samples
        end_idx = min(i + batch_size - 1, n_samples)
        batch_images = images[i:end_idx]
        batch_labels = labels[i:end_idx]
        batch_tensor = cat(batch_images..., dims=4)
        
        try
            batch_labels_onehot = Flux.onehotbatch(batch_labels, label_range)
            push!(batches, (batch_tensor, batch_labels_onehot))
            println("   📦 Batch $(div(i-1, batch_size)+1): $(length(batch_labels)) amostras")
        catch e
            println("❌ Erro criando batch $i-$end_idx: $e")
            continue
        end
    end
    
    return batches
end

# CNN architecture optimized for face recognition (mantida igual)
function create_pretrain_cnn_model(num_classes::Int, input_size::Tuple{Int, Int} = CNNCheckinCore.IMG_SIZE)
    # Calculate final feature size after convolutions and pooling
    final_size = div(div(div(div(input_size[1], 2), 2), 2), 2)
    final_features = 256 * final_size * final_size
    
    return Chain(
        # Feature extraction layers
        Conv((3, 3), 3 => 64, relu, pad=1),
        BatchNorm(64),
        Dropout(0.1),
        MaxPool((2, 2)),
        
        Conv((3, 3), 64 => 128, relu, pad=1),
        BatchNorm(128),
        Dropout(0.1),
        MaxPool((2, 2)),
        
        Conv((3, 3), 128 => 256, relu, pad=1),
        BatchNorm(256),
        Dropout(0.15),
        MaxPool((2, 2)),
        
        Conv((3, 3), 256 => 256, relu, pad=1),
        BatchNorm(256),
        Dropout(0.15),
        MaxPool((2, 2)),
        
        # Classifier layers
        Flux.flatten,
        Dense(final_features, 512, relu),
        Dropout(0.4),
        Dense(512, 256, relu),
        Dropout(0.3),
        Dense(256, num_classes)
    )
end

# Calculate accuracy (mantida igual)
function pretrain_accuracy(model, data_loader)
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
            println("❌ Erro calculando acurácia: $e")
            continue
        end
    end
    return total > 0 ? correct / total : 0.0
end

# Pre-training function (mantida igual)
function pretrain_model(model, train_data, val_data, epochs, learning_rate)
    println("🚀 Iniciando pré-treino...")
    
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
        
        # Training phase
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
                println("❌ Erro no batch de treino epoch $epoch: $e")
                continue
            end
        end
        
        avg_loss = num_batches > 0 ? epoch_loss / num_batches : 0.0
        val_acc = pretrain_accuracy(model, val_data)
        
        push!(train_losses, avg_loss)
        push!(val_accuracies, val_acc)
        
        if val_acc > best_val_acc
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
        else
            patience_counter += 1
        end
        
        if epoch % 3 == 0 || epoch == 1
            println("📈 Epoch $epoch/$epochs - Loss: $(round(avg_loss, digits=4)) - Val Acc: $(round(val_acc*100, digits=2))% - Best: $(round(best_val_acc*100, digits=2))% (Epoch $best_epoch)")
        end
        
        # Early stopping
        if patience_counter >= patience_limit
            println("ℹ️ Parada antecipada - sem melhoria por $patience_limit epochs")
            break
        end
        
        # Stop if achieving good accuracy
        if val_acc >= 0.88
            println("🎯 Boa acurácia alcançada!")
            break
        end
    end
    
    return train_losses, val_accuracies, best_val_acc, best_epoch
end

# Save pre-trained model and configuration - VERSÃO MODIFICADA COM TOML
function save_pretrained_model_with_toml(model, person_names, model_filepath::String, 
                                        config_filepath::String, training_info::Dict)
    println("💾 Salvando modelo pré-treinado com suporte TOML...")
    
    # 1. Salvar pesos e vieses em formato TOML (NOVO)
    weights_saved = save_pretrained_weights_toml(model, person_names, training_info, WEIGHTS_TOML_PATH)
    
    # 2. Salvar modelo JLD2 (mantido para compatibilidade)
    model_data = Dict(
        "model_state" => model,
        "person_names" => person_names,
        "model_type" => "pretrained",
        "timestamp" => string(Dates.now()),
        "training_info" => training_info
    )
    
    jld2_saved = false
    try
        jldsave(model_filepath; model_data=model_data)
        println("✅ Modelo JLD2 salvo em: $model_filepath")
        jld2_saved = true
    catch e
        println("❌ Erro ao salvar modelo JLD2: $e")
    end
    
    # 3. Salvar dados do modelo em TOML (mantido)
    model_data_saved = CNNCheckinCore.save_model_data_toml(model, person_names, 
                                                          CNNCheckinCore.MODEL_DATA_TOML_PATH)
    
    # 4. Criar e salvar configuração (mantido)
    config = CNNCheckinCore.create_default_config()
    config["model"]["num_classes"] = length(person_names)
    config["model"]["augmentation_used"] = training_info["augmentation_used"]
    config["training"]["epochs_trained"] = training_info["epochs_trained"]
    config["training"]["final_accuracy"] = training_info["final_accuracy"]
    config["training"]["best_epoch"] = training_info["best_epoch"]
    config["data"]["person_names"] = person_names
    config["data"]["timestamp"] = string(Dates.now())
    
    # Adicionar referência ao arquivo de pesos TOML
    config["weights"] = Dict(
        "toml_file" => WEIGHTS_TOML_PATH,
        "format" => "accumulated_trainings",
        "supports_incremental" => true
    )
    
    config_saved = CNNCheckinCore.save_config(config, config_filepath)
    
    # Status do salvamento
    println("\n📋 Status do Salvamento:")
    println("   ✅ Pesos TOML: $(weights_saved ? "Salvo" : "Falhou")")
    println("   $(jld2_saved ? "✅" : "❌") Modelo JLD2: $(jld2_saved ? "Salvo" : "Falhou")")
    println("   $(model_data_saved ? "✅" : "❌") Dados TOML: $(model_data_saved ? "Salvo" : "Falhou")")
    println("   $(config_saved ? "✅" : "❌") Config TOML: $(config_saved ? "Salvo" : "Falhou")")
    
    return weights_saved && config_saved
end

# Main pre-training function - VERSÃO MODIFICADA
function pretrain_command()
    println("🧠 Sistema de Reconhecimento Facial - Modo Pré-treino com TOML")
    println("📝 Arquivo de pesos: $WEIGHTS_TOML_PATH")
    
    start_time = time()
    
    try
        # Load pre-training data
        people_data, person_names = load_pretrain_data(CNNCheckinCore.TRAIN_DATA_PATH; use_augmentation=true)
        if length(people_data) == 0
            error("Nenhum dado de treino encontrado!")
        end
        
        num_classes = length(person_names)
        println("👥 Total de pessoas: $num_classes")
        total_images = sum(length(person.images) for person in people_data)
        println("📸 Total de imagens (com augmentação): $total_images")
        
        # Verificar se já existem treinamentos anteriores
        if isfile(WEIGHTS_TOML_PATH)
            println("\n📚 Treinamentos anteriores encontrados:")
            list_saved_trainings(WEIGHTS_TOML_PATH)
            println()
        else
            println("\n🆕 Primeiro treinamento - criando arquivo de pesos")
        end
        
        # Create datasets and batches
        (train_images, train_labels), (val_images, val_labels) = create_pretrain_datasets(people_data)
        train_batches = create_pretrain_batches(train_images, train_labels, CNNCheckinCore.BATCH_SIZE)
        val_batches = create_pretrain_batches(val_images, val_labels, CNNCheckinCore.BATCH_SIZE)
        
        if length(train_batches) == 0
            error("Não foi possível criar batches de treino!")
        end
        
        # Create and train model
        println("🗃️ Criando modelo CNN...")
        model = create_pretrain_cnn_model(num_classes)
        train_losses, val_accuracies, best_val_acc, best_epoch = pretrain_model(model, train_batches, val_batches, 
                                                                               CNNCheckinCore.PRETRAIN_EPOCHS, 
                                                                               CNNCheckinCore.LEARNING_RATE)
        
        end_time = time()
        duration_minutes = (end_time - start_time) / 60
        
        # Prepare training info
        training_info = Dict(
            "epochs_trained" => length(val_accuracies),
            "final_accuracy" => best_val_acc,
            "best_epoch" => best_epoch,
            "total_training_images" => length(train_images),
            "total_validation_images" => length(val_images),
            "augmentation_used" => true,
            "duration_minutes" => duration_minutes,
            "model_architecture" => "CNN_FaceRecognition_v1",
            "learning_rate" => CNNCheckinCore.LEARNING_RATE,
            "batch_size" => CNNCheckinCore.BATCH_SIZE
        )
        
        println("\n🎉 Pré-treino concluído!")
        println("📊 Resultados:")
        println("   - Melhor acurácia: $(round(best_val_acc*100, digits=2))% (Epoch $best_epoch)")
        println("   - Epochs treinadas: $(training_info["epochs_trained"])/$(CNNCheckinCore.PRETRAIN_EPOCHS)")
        println("   - Duração: $(round(duration_minutes, digits=1)) minutos")
        
        # Save model and configuration with TOML support
        success = save_pretrained_model_with_toml(model, person_names, CNNCheckinCore.MODEL_PATH, 
                                                 CNNCheckinCore.CONFIG_PATH, training_info)
        
        if success
            println("\n✅ Sistema pré-treinado salvo com sucesso!")
            println("\n📁 Arquivos gerados:")
            println("   - Configuração: $(CNNCheckinCore.CONFIG_PATH)")
            println("   - Modelo JLD2: $(CNNCheckinCore.MODEL_PATH)")
            println("   - Dados TOML: $(CNNCheckinCore.MODEL_DATA_TOML_PATH)")
            println("   - Pesos TOML: $WEIGHTS_TOML_PATH")
            
            # Mostrar estatísticas dos arquivos de peso
            println("\n📈 Resumo dos Treinamentos:")
            list_saved_trainings(WEIGHTS_TOML_PATH)
        else
            println("❌ Modelo treinado mas alguns arquivos falharam ao salvar")
        end
        
        return success
        
    catch e
        println("❌ Erro durante pré-treino: $e")
        return false
    end
end

# Função para comparar treinamentos (NOVA)
function compare_trainings_command()
    println("🔍 Comparar Treinamentos")
    
    if !isfile(WEIGHTS_TOML_PATH)
        println("❌ Arquivo de pesos não encontrado: $WEIGHTS_TOML_PATH")
        return false
    end
    
    # Listar treinamentos disponíveis
    list_saved_trainings(WEIGHTS_TOML_PATH)
    
    print("\n Digite o ID do primeiro treinamento: ")
    id1 = strip(readline())
    
    print("Digite o ID do segundo treinamento: ")
    id2 = strip(readline())
    
    if isempty(id1) || isempty(id2)
        println("❌ IDs não podem estar vazios!")
        return false
    end
    
    return compare_training_weights(WEIGHTS_TOML_PATH, id1, id2)
end

# Função para carregar modelo específico (NOVA)
function load_specific_training_command()
    println("📂 Carregar Treinamento Específico")
    
    if !isfile(WEIGHTS_TOML_PATH)
        println("❌ Arquivo de pesos não encontrado: $WEIGHTS_TOML_PATH")
        return nothing
    end
    
    # Listar treinamentos disponíveis
    list_saved_trainings(WEIGHTS_TOML_PATH)
    
    print("\nDigite o ID do treinamento para carregar: ")
    training_id = strip(readline())
    
    if isempty(training_id)
        println("❌ ID não pode estar vazio!")
        return nothing
    end
    
    training_data = load_weights_from_toml(WEIGHTS_TOML_PATH, training_id)
    
    if training_data !== nothing
        println("✅ Treinamento carregado com sucesso!")
        
        # Criar modelo com a arquitetura correta
        person_names = training_data["metadata"]["person_names"]
        num_classes = length(person_names)
        model = create_pretrain_cnn_model(num_classes)
        
        # Tentar reconstruir o modelo (experimental)
        reconstructed_model = reconstruct_model_from_weights(training_data, model)
        
        return (reconstructed_model, person_names, training_data)
    end
    
    return nothing
end

# Menu principal para gerenciamento de treinamentos (NOVO)
function training_management_menu()
    println("\n🎛️  MENU DE GERENCIAMENTO DE TREINAMENTOS")
    println("=" * 50)
    println("1 - Executar novo treinamento")
    println("2 - Listar treinamentos salvos")
    println("3 - Comparar dois treinamentos")
    println("4 - Carregar treinamento específico")
    println("5 - Sair")
    
    print("Escolha uma opção: ")
    choice = strip(readline())
    
    if choice == "1"
        return pretrain_command()
    elseif choice == "2"
        list_saved_trainings(WEIGHTS_TOML_PATH)
        return true
    elseif choice == "3"
        return compare_trainings_command()
    elseif choice == "4"
        result = load_specific_training_command()
        return result !== nothing
    elseif choice == "5"
        println("👋 Saindo...")
        return false
    else
        println("❌ Opção inválida!")
        return training_management_menu()
    end
end

# Export the main function
export pretrain_command, training_management_menu, compare_trainings_command, 
       load_specific_training_command, WEIGHTS_TOML_PATH