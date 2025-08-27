# projeto: cnncheckin
# file: cnncheckin/src/cnncheckin_pretrain.jl
using Flux
using Statistics
using Random
using JLD2
using Dates

include("cnncheckin_core.jl")
using .CNNCheckinCore

# Function to load initial training data
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

# Create balanced datasets for pre-training
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

# Create training batches
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

# CNN architecture optimized for face recognition
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

# Calculate accuracy
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

# Pre-training function
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
            println("⏹️ Parada antecipada - sem melhoria por $patience_limit epochs")
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

# Save pre-trained model and configuration
function save_pretrained_model(model, person_names, model_filepath::String, 
                              config_filepath::String, training_info::Dict)
    println("💾 Salvando modelo pré-treinado...")
    
    # Save model weights
    model_data = Dict(
        "model_state" => model,
        "person_names" => person_names,
        "model_type" => "pretrained",
        "timestamp" => string(Dates.now()),
        "training_info" => training_info
    )
    
    try
        jldsave(model_filepath; model_data=model_data)
        println("✅ Modelo salvo em: $model_filepath")
    catch e
        println("❌ Erro ao salvar modelo: $e")
        return false
    end
    
    # Save model data to TOML
    model_data_saved = CNNCheckinCore.save_model_data_toml(model, person_names, 
                                                          CNNCheckinCore.MODEL_DATA_TOML_PATH)
    
    # Create and save configuration
    config = CNNCheckinCore.create_default_config()
    config["model"]["num_classes"] = length(person_names)
    config["model"]["augmentation_used"] = training_info["augmentation_used"]
    config["training"]["epochs_trained"] = training_info["epochs_trained"]
    config["training"]["final_accuracy"] = training_info["final_accuracy"]
    config["training"]["best_epoch"] = training_info["best_epoch"]
    config["data"]["person_names"] = person_names
    config["data"]["timestamp"] = string(Dates.now())
    
    config_saved = CNNCheckinCore.save_config(config, config_filepath)
    
    return config_saved && model_data_saved
end

# Main pre-training function
function pretrain_command()
    println("🧠 Sistema de Reconhecimento Facial - Modo Pré-treino")
    
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
        
        # Create datasets and batches
        (train_images, train_labels), (val_images, val_labels) = create_pretrain_datasets(people_data)
        train_batches = create_pretrain_batches(train_images, train_labels, CNNCheckinCore.BATCH_SIZE)
        val_batches = create_pretrain_batches(val_images, val_labels, CNNCheckinCore.BATCH_SIZE)
        
        if length(train_batches) == 0
            error("Não foi possível criar batches de treino!")
        end
        
        # Create and train model
        println("🏗️ Criando modelo CNN...")
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
            "duration_minutes" => duration_minutes
        )
        
        println("\n🎉 Pré-treino concluído!")
        println("📊 Resultados:")
        println("   - Melhor acurácia: $(round(best_val_acc*100, digits=2))% (Epoch $best_epoch)")
        println("   - Epochs treinadas: $(training_info["epochs_trained"])/$(CNNCheckinCore.PRETRAIN_EPOCHS)")
        println("   - Duração: $(round(duration_minutes, digits=1)) minutos")
        
        # Save model and configuration
        success = save_pretrained_model(model, person_names, CNNCheckinCore.MODEL_PATH, 
                                       CNNCheckinCore.CONFIG_PATH, training_info)
        
        if success
            println("Sistema pré-treinado salvo com sucesso!")
            println("\nArquivos gerados:")
            println("   - Configuração: $(CNNCheckinCore.CONFIG_PATH)")
            println("   - Modelo: $(CNNCheckinCore.MODEL_PATH)")
            println("   - Dados do modelo: $(CNNCheckinCore.MODEL_DATA_TOML_PATH)")
            println("\nPróximos passos:")
            println("   1. Para treino incremental: julia cnncheckin_incremental.jl")
            println("   2. Para identificação: julia cnncheckin_identify.jl <caminho_da_imagem>")
        else
            println("Modelo treinado mas alguns arquivos falharam ao salvar")
        end
        
        return success
        
    catch e
        println("Erro durante pré-treino: $e")
        return false
    end
end

# Execute command if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    success = pretrain_command()
    if success
        println("Pré-treino concluído com sucesso!")
    else
        println("Pré-treino falhou")
    end
end



# julia cnncheckin_pretrain.jl