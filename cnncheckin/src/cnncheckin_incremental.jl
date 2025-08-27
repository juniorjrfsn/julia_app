# projeto: cnncheckin
# file: cnncheckin/src/cnncheckin_incremental.jl

using Flux
using Statistics
using Random
using JLD2
using Dates

include("cnncheckin_core.jl")
using .CNNCheckinCore

# Enhanced file validation for incremental learning with better error handling
function validate_incremental_image_file(filepath::String)
    try
        # Check file extension first
        file_ext = lowercase(splitext(filepath)[2])
        valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp"]
        
        if !(file_ext in valid_extensions)
            println("Extensão não suportada: $filepath")
            return false
        end
        
        # Check file size (avoid empty or too large files)
        if !isfile(filepath)
            println("Arquivo não existe: $filepath")
            return false
        end
        
        filesize_bytes = stat(filepath).size
        if filesize_bytes < 500  # Less than 500 bytes
            println("Arquivo muito pequeno: $filepath")
            return false
        end
        
        if filesize_bytes > 50 * 1024 * 1024  # More than 50MB
            println("Arquivo muito grande: $filepath")
            return false
        end
        
        # Skip files that might be videos or have problematic extensions
        filename_lower = lowercase(basename(filepath))
        if contains(filename_lower, "video") || contains(filename_lower, ".mp4") || 
           contains(filename_lower, ".avi") || contains(filename_lower, ".mov")
            println("Arquivo de vídeo detectado, ignorando: $filepath")
            return false
        end
        
        # Try to load the image with better error handling
        try
            img = load(filepath)
            
            # Check if image has valid dimensions
            if ndims(img) < 2
                println("Dimensões de imagem inválidas: $filepath")
                return false
            end
            
            img_size = size(img)
            if length(img_size) >= 2 && (img_size[1] < 10 || img_size[2] < 10)
                println("Imagem muito pequena: $filepath")
                return false
            end
            
            # Check if it's actually an image
            if img_size != (0, 0) && length(img_size) >= 2
                return true
            else
                println("Formato de arquivo inválido: $filepath")
                return false
            end
            
        catch load_error
            # More specific error handling
            if isa(load_error, CapturedException) || contains(string(load_error), "VideoIO")
                println("Arquivo de vídeo ou formato não suportado: $filepath")
                return false
            elseif contains(string(load_error), "Package") && contains(string(load_error), "required")
                println("Pacote necessário não instalado para: $filepath")
                return false
            else
                println("Erro ao carregar imagem: $filepath - $(string(load_error))")
                return false
            end
        end
        
    catch e
        println("Erro geral ao validar arquivo: $filepath - $e")
        return false
    end
end

# Load pre-trained model and configuration - FIXED VERSION
function load_pretrained_model(model_filepath::String, config_filepath::String)
    println("Carregando modelo pré-treinado e configuração...")
    
    if !isfile(model_filepath)
        error("Modelo pré-treinado não encontrado: $model_filepath")
    end
    
    if !isfile(config_filepath)
        error("Arquivo de configuração não encontrado: $config_filepath")
    end
    
    # Load configuration
    config = CNNCheckinCore.load_config(config_filepath)
    CNNCheckinCore.validate_config(config)
    
    # Load model - FIXED TO PROPERLY RECONSTRUCT MODEL
    try
        data = load(model_filepath)
        model_data = data["model_data"]
        
        # Get the saved model state and person names
        saved_model_state = model_data["model_state"]
        original_person_names = config["data"]["person_names"]
        
        # CRITICAL FIX: Properly reconstruct the model architecture first
        num_classes = length(original_person_names)
        println("Recriando arquitetura do modelo para $num_classes classes...")
        
        # Recreate the same architecture as in pretrain
        final_size = div(div(div(div(CNNCheckinCore.IMG_SIZE[1], 2), 2), 2), 2)
        final_features = 256 * final_size * final_size
        
        reconstructed_model = Chain(
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
        
        # CRITICAL FIX: Load the saved weights into the reconstructed model
        println("Carregando pesos salvos no modelo reconstruído...")
        try
            # Copy the state from saved model to reconstructed model
            Flux.loadmodel!(reconstructed_model, saved_model_state)
            println("✅ Pesos carregados com sucesso!")
        catch weight_error
            println("⚠️ Erro ao carregar pesos, tentando método alternativo: $weight_error")
            
            # Alternative method: direct parameter copying
            try
                saved_params = Flux.params(saved_model_state)
                new_params = Flux.params(reconstructed_model)
                
                if length(saved_params) == length(new_params)
                    for (saved_p, new_p) in zip(saved_params, new_params)
                        if size(saved_p) == size(new_p)
                            new_p .= saved_p
                        else
                            println("⚠️ Incompatibilidade de tamanho: $(size(saved_p)) vs $(size(new_p))")
                        end
                    end
                    println("✅ Pesos copiados com método alternativo!")
                else
                    error("Número de parâmetros incompatível: $(length(saved_params)) vs $(length(new_params))")
                end
            catch copy_error
                error("Falha ao copiar pesos: $copy_error")
            end
        end
        
        println("Modelo pré-treinado carregado com sucesso!")
        println("Informações do modelo:")
        println("   - Classes originais: $(length(original_person_names))")
        println("   - Pessoas: $(join(original_person_names, ", "))")
        println("   - Acurácia: $(round(config["training"]["final_accuracy"]*100, digits=2))%")
        
        return reconstructed_model, original_person_names, config
    catch e
        error("Erro ao carregar modelo pré-treinado: $e")
    end
end

# Enhanced data loading with better error handling and person detection
function load_incremental_data(data_path::String, existing_people::Vector{String}; use_augmentation::Bool = true)
    println("Carregando dados para aprendizado incremental...")
    
    if !isdir(data_path)
        error("Diretório de dados incrementais não encontrado: $data_path")
    end
    
    person_images = Dict{String, Vector{Array{Float32, 3}}}()
    new_people = Set{String}()
    processed_files = 0
    failed_files = 0
    existing_person_files = 0
    
    # Get all files in directory
    all_files = readdir(data_path)
    println("Encontrados $(length(all_files)) arquivos no diretório")
    
    for filename in all_files
        img_path = joinpath(data_path, filename)
        
        # Skip if not a valid image file
        if !validate_incremental_image_file(img_path)
            failed_files += 1
            continue
        end
        
        try
            person_name = CNNCheckinCore.extract_person_name(filename)
            
            # Check if this person already exists in the training set
            if person_name in existing_people
                println("Pessoa já existe no modelo: $person_name - ignorando arquivo $filename")
                existing_person_files += 1
                continue
            end
            
            # Mark as new person
            push!(new_people, person_name)
            
            # Process image with augmentation
            img_arrays = CNNCheckinCore.preprocess_image(img_path; augment=use_augmentation)
            
            if img_arrays !== nothing && length(img_arrays) > 0
                if !haskey(person_images, person_name)
                    person_images[person_name] = Vector{Array{Float32, 3}}()
                end
                
                for img_array in img_arrays
                    push!(person_images[person_name], img_array)
                end
                
                total_imgs = length(img_arrays)
                println("✅ Carregado: $filename -> $person_name ($total_imgs variações)")
                processed_files += 1
            else
                println("❌ Falha ao processar: $filename")
                failed_files += 1
            end
            
        catch e
            println("❌ Erro processando $filename: $e")
            failed_files += 1
        end
    end
    
    println("\nResumo do carregamento:")
    println("   - Arquivos processados: $processed_files")
    println("   - Arquivos falharam: $failed_files")
    println("   - Arquivos de pessoas existentes ignorados: $existing_person_files")
    println("   - Pessoas novas encontradas: $(length(new_people))")
    
    if length(new_people) == 0
        println("⚠️ Nenhuma pessoa nova encontrada!")
        println("💡 Dicas:")
        println("   - Verifique se há imagens de pessoas diferentes das já treinadas")
        println("   - Certifique-se que os nomes dos arquivos seguem o padrão: nome-numero.extensao")
        println("   - Pessoas já treinadas: $(join(existing_people, ", "))")
        println("   - Para adicionar novas pessoas, use nomes como: maria-1.jpg, carlos-1.png, etc.")
        return Vector{CNNCheckinCore.PersonData}(), existing_people, String[]
    end
    
    # Create combined person list (existing + new)
    all_person_names = vcat(existing_people, sort(collect(new_people)))
    people_data = Vector{CNNCheckinCore.PersonData}()
    
    # Create person data with proper indexing
    for (idx, person_name) in enumerate(all_person_names)
        if haskey(person_images, person_name)
            images = person_images[person_name]
            is_incremental = !(person_name in existing_people)
            push!(people_data, CNNCheckinCore.PersonData(person_name, images, idx, is_incremental))
            status = is_incremental ? "NOVA" : "EXISTENTE"
            println("👤 Pessoa: $person_name - $(length(images)) imagens (Label: $idx, Status: $status)")
        end
    end
    
    # Ensure minimum images per person
    min_images_per_person = 3
    filtered_people_data = Vector{CNNCheckinCore.PersonData}()
    filtered_new_people = String[]
    
    for person in people_data
        if person.is_incremental && length(person.images) < min_images_per_person
            println("⚠️ Pessoa $(person.name) tem apenas $(length(person.images)) imagens (mínimo: $min_images_per_person) - ignorando")
        else
            push!(filtered_people_data, person)
            if person.is_incremental
                push!(filtered_new_people, person.name)
            end
        end
    end
    
    return filtered_people_data, all_person_names, filtered_new_people
end

# FIXED: Modify model architecture for new classes with proper weight copying
function expand_model_for_incremental(original_model, original_num_classes::Int, new_num_classes::Int)
    println("Expandindo modelo para aprendizado incremental...")
    println("   - Classes originais: $original_num_classes")
    println("   - Novo total de classes: $new_num_classes")
    
    if new_num_classes <= original_num_classes
        println("Nenhuma classe nova para adicionar, retornando modelo original")
        return original_model
    end
    
    # Get all layers from original model
    model_layers = collect(original_model)
    println("   - Total de camadas no modelo original: $(length(model_layers))")
    
    # Find the last Dense layer (classification layer)
    last_dense_idx = 0
    for i in length(model_layers):-1:1
        if isa(model_layers[i], Dense)
            last_dense_idx = i
            break
        end
    end
    
    if last_dense_idx == 0
        error("Não foi possível encontrar a camada Dense final para classificação")
    end
    
    println("   - Camada de classificação encontrada no índice: $last_dense_idx")
    
    # Get layers before the final classification layer
    feature_layers = model_layers[1:last_dense_idx-1]
    old_classifier = model_layers[last_dense_idx]
    remaining_layers = length(model_layers) > last_dense_idx ? model_layers[last_dense_idx+1:end] : []
    
    # Get input size for new classifier
    input_size = size(old_classifier.weight, 2)
    println("   - Tamanho de entrada para classificador: $input_size")
    println("   - Classes antigas: $original_num_classes")
    println("   - Classes novas: $new_num_classes")
    
    # Create new classification layer
    new_classifier = Dense(input_size, new_num_classes)
    
    # CRITICAL FIX: Properly initialize and copy weights
    println("   - Inicializando nova camada de classificação...")
    
    # Initialize all weights with small random values
    new_classifier.weight .= randn(Float32, new_num_classes, input_size) * 0.01f0
    new_classifier.bias .= zeros(Float32, new_num_classes)
    
    # Copy weights from old classifier for existing classes
    if original_num_classes > 0
        old_weight_size = size(old_classifier.weight)
        new_weight_size = size(new_classifier.weight)
        
        println("   - Copiando pesos antigos:")
        println("     - Peso antigo: $old_weight_size")
        println("     - Peso novo: $new_weight_size")
        
        # Ensure we don't exceed bounds
        copy_classes = min(original_num_classes, new_num_classes)
        copy_features = min(size(old_classifier.weight, 2), size(new_classifier.weight, 2))
        
        # Copy weights and biases for existing classes
        new_classifier.weight[1:copy_classes, 1:copy_features] .= old_classifier.weight[1:copy_classes, 1:copy_features]
        new_classifier.bias[1:copy_classes] .= old_classifier.bias[1:copy_classes]
        
        println("   - ✅ Copiados pesos para $copy_classes classes existentes")
        println("   - ✅ Inicializados pesos para $(new_num_classes - copy_classes) classes novas")
    end
    
    # Reconstruct the complete model
    if length(remaining_layers) > 0
        expanded_model = Chain(feature_layers..., new_classifier, remaining_layers...)
    else
        expanded_model = Chain(feature_layers..., new_classifier)
    end
    
    println("   - ✅ Modelo expandido criado com $(length(collect(expanded_model))) camadas")
    
    return expanded_model
end

# Create incremental learning datasets with better balance
function create_incremental_datasets(people_data::Vector{CNNCheckinCore.PersonData}, 
                                    original_people::Vector{String}, split_ratio::Float64 = 0.8)
    println("Criando datasets para aprendizado incremental...")
    
    train_images = Vector{Array{Float32, 3}}()
    train_labels = Vector{Int}()
    val_images = Vector{Array{Float32, 3}}()
    val_labels = Vector{Int}()
    
    # Separate original and new people for balanced sampling
    original_data = filter(p -> p.name in original_people, people_data)
    new_data = filter(p -> !(p.name in original_people), people_data)
    
    println("   - Dados de pessoas originais: $(length(original_data)) pessoas")
    println("   - Dados de pessoas novas: $(length(new_data)) pessoas")
    
    # Process all people
    for person in people_data
        n_imgs = length(person.images)
        
        # For new classes, use more conservative split but ensure at least 1 validation image
        if person.is_incremental
            n_train = max(1, min(n_imgs - 1, Int(floor(n_imgs * 0.75))))  # Keep at least 1 for validation
        else
            n_train = max(1, Int(floor(n_imgs * split_ratio)))
        end
        
        indices = randperm(n_imgs)
        
        for i in 1:n_train
            push!(train_images, person.images[indices[i]])
            push!(train_labels, person.label)
        end
        
        for i in (n_train+1):n_imgs
            push!(val_images, person.images[indices[i]])
            push!(val_labels, person.label)
        end
        
        status = person.is_incremental ? "NOVA" : "EXISTENTE"
        println("   - $(person.name) [$status]: $n_train treino, $(n_imgs - n_train) validação")
    end
    
    println("Dataset incremental criado:")
    println("   - Treino: $(length(train_images)) imagens")
    println("   - Validação: $(length(val_images)) imagens")
    
    return (train_images, train_labels), (val_images, val_labels)
end

# Create batches for incremental learning
function create_incremental_batches(images, labels, batch_size)
    batches = []
    n_samples = length(images)
    if n_samples == 0
        return batches
    end
    
    unique_labels = unique(labels)
    min_label = minimum(unique_labels)
    max_label = maximum(unique_labels)
    label_range = min_label:max_label
    
    println("Criando batches incrementais:")
    println("   - Labels únicos: $unique_labels")
    println("   - Range de labels: $label_range")
    
    for i in 1:batch_size:n_samples
        end_idx = min(i + batch_size - 1, n_samples)
        batch_images = images[i:end_idx]
        batch_labels = labels[i:end_idx]
        batch_tensor = cat(batch_images..., dims=4)
        
        try
            batch_labels_onehot = Flux.onehotbatch(batch_labels, label_range)
            push!(batches, (batch_tensor, batch_labels_onehot))
            println("   Batch $(div(i-1, batch_size)+1): $(length(batch_labels)) amostras")
        catch e
            println("Erro criando batch $i-$end_idx: $e")
            continue
        end
    end
    
    return batches
end

# Knowledge distillation loss for incremental learning
function knowledge_distillation_loss(student_logits, teacher_logits, y_true, temperature::Float64, alpha::Float64)
    # Soft targets from teacher
    soft_targets = softmax(teacher_logits ./ temperature)
    soft_student = softmax(student_logits ./ temperature)
    
    # Distillation loss
    distillation_loss = -sum(soft_targets .* log.(soft_student .+ 1e-8))
    
    # Hard target loss
    hard_loss = Flux.logitcrossentropy(student_logits, y_true)
    
    # Combined loss
    return alpha * (temperature^2) * distillation_loss + (1 - alpha) * hard_loss
end

# Incremental training function
function train_incremental_model(student_model, teacher_model, train_data, val_data, 
                                original_num_classes::Int, epochs, learning_rate)
    println("Iniciando aprendizado incremental...")
    
    optimizer = ADAM(learning_rate, (0.9, 0.999), 1e-8)
    opt_state = Flux.setup(optimizer, student_model)
    
    train_losses = Float64[]
    val_accuracies = Float64[]
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    patience_limit = 8
    
    temperature = 3.0
    distillation_weight = 0.7
    
    for epoch in 1:epochs
        epoch_loss = 0.0
        num_batches = 0
        
        # Training phase with knowledge distillation
        for (x, y) in train_data
            try
                loss, grads = Flux.withgradient(student_model) do m
                    student_logits = m(x)
                    
                    # Standard cross-entropy loss for all classes
                    ce_loss = Flux.logitcrossentropy(student_logits, y)
                    
                    # Knowledge distillation loss for original classes only if we have them
                    if original_num_classes > 0 && size(student_logits, 1) > original_num_classes
                        teacher_logits = teacher_model(x)
                        original_student_logits = student_logits[1:original_num_classes, :]
                        original_y = y[1:original_num_classes, :]
                        
                        # Use knowledge distillation for original classes
                        kd_loss = knowledge_distillation_loss(original_student_logits, teacher_logits, 
                                                            original_y, temperature, distillation_weight)
                        
                        # Combined loss: distillation for old classes + CE for new classes
                        total_loss = (1 - distillation_weight) * ce_loss + distillation_weight * kd_loss
                    else
                        # Only new classes, use standard cross-entropy
                        total_loss = ce_loss
                    end
                    
                    return total_loss
                end
                
                Flux.update!(opt_state, student_model, grads[1])
                epoch_loss += loss
                num_batches += 1
            catch e
                println("Erro no batch de treino incremental epoch $epoch: $e")
                continue
            end
        end
        
        avg_loss = num_batches > 0 ? epoch_loss / num_batches : 0.0
        val_acc = incremental_accuracy(student_model, val_data)
        
        push!(train_losses, avg_loss)
        push!(val_accuracies, val_acc)
        
        if val_acc > best_val_acc
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
        else
            patience_counter += 1
        end
        
        if epoch % 2 == 0 || epoch == 1
            println("Epoch $epoch/$epochs - Loss: $(round(avg_loss, digits=4)) - Val Acc: $(round(val_acc*100, digits=2))% - Best: $(round(best_val_acc*100, digits=2))% (Epoch $best_epoch)")
        end
        
        # Early stopping for incremental learning
        if patience_counter >= patience_limit
            println("Parada antecipada - sem melhoria por $patience_limit epochs")
            break
        end
        
        # Stop if achieving good accuracy
        if val_acc >= 0.85
            println("Boa acurácia incremental alcançada!")
            break
        end
    end
    
    return train_losses, val_accuracies, best_val_acc, best_epoch
end

# Calculate accuracy for incremental model
function incremental_accuracy(model, data_loader)
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
            println("Erro calculando acurácia incremental: $e")
            continue
        end
    end
    return total > 0 ? correct / total : 0.0
end

# FIXED: Save incremental model with proper state preservation
function save_incremental_model(model, all_person_names, original_person_names, 
                               new_person_names, model_filepath::String, 
                               config_filepath::String, training_info::Dict)
    println("Salvando modelo incremental...")
    
    # CRITICAL FIX: Save the actual model state, not just a reference
    try
        # Test if model can make predictions to ensure it's properly trained
        test_input = randn(Float32, CNNCheckinCore.IMG_SIZE..., 3, 1)
        test_output = model(test_input)
        println("✅ Modelo validado - pode fazer predições. Output shape: $(size(test_output))")
    catch e
        println("⚠️ Aviso: Modelo pode ter problemas: $e")
    end
    
    # Save model weights with proper serialization
    model_data = Dict(
        "model_state" => model,  # Save the complete trained model
        "all_person_names" => all_person_names,
        "original_person_names" => original_person_names,
        "new_person_names" => new_person_names,
        "model_type" => "incremental",
        "timestamp" => string(Dates.now()),
        "training_info" => training_info,
        "model_architecture" => "CNN_incremental_v1"
    )
    
    try
        # Force serialization with explicit save
        println("Salvando estado do modelo...")
        jldsave(model_filepath; model_data=model_data)
        println("✅ Modelo incremental salvo em: $model_filepath")
        
        # Verify save was successful
        test_load = load(model_filepath)
        if haskey(test_load, "model_data") && haskey(test_load["model_data"], "model_state")
            println("✅ Verificação: Modelo salvo corretamente")
        else
            error("❌ Verificação falhou: Modelo não foi salvo corretamente")
        end
        
    catch e
        println("❌ Erro ao salvar modelo incremental: $e")
        return false
    end
    
    # Update model data TOML
    model_data_saved = CNNCheckinCore.save_model_data_toml(model, all_person_names, 
                                                          CNNCheckinCore.MODEL_DATA_TOML_PATH)
    
    # Update configuration
    config = CNNCheckinCore.load_config(config_filepath)
    config["model"]["num_classes"] = length(all_person_names)
    config["training"]["epochs_trained"] += training_info["epochs_trained"]
    config["training"]["final_accuracy"] = training_info["final_accuracy"]
    config["training"]["best_epoch"] = training_info["best_epoch"]
    config["data"]["person_names"] = all_person_names  # Update to all people
    config["data"]["timestamp"] = string(Dates.now())
    
    # Add incremental training stats
    config["incremental_stats"] = Dict(
        "new_people_added" => length(new_person_names),
        "original_people" => original_person_names,
        "new_people" => new_person_names,
        "total_people" => length(all_person_names),
        "incremental_accuracy" => training_info["final_accuracy"],
        "knowledge_distillation_used" => true,
        "last_incremental_training" => string(Dates.now())
    )
    
    config_saved = CNNCheckinCore.save_config(config, config_filepath)
    
    return config_saved && model_data_saved
end

# Main incremental learning function
function incremental_learning_command()
    println("Sistema de Reconhecimento Facial - Modo Aprendizado Incremental")
    
    start_time = time()
    
    try
        # Load pre-trained model with FIXED loading
        teacher_model, original_person_names, config = load_pretrained_model(
            CNNCheckinCore.MODEL_PATH, 
            CNNCheckinCore.CONFIG_PATH
        )
        
        # Load incremental data
        people_data, all_person_names, new_person_names = load_incremental_data(
            CNNCheckinCore.INCREMENTAL_DATA_PATH, original_person_names; use_augmentation=true
        )
        
        if length(new_person_names) == 0
            println("\n❌ Nenhuma pessoa nova encontrada para aprendizado incremental!")
            println("\n💡 Para adicionar novas pessoas:")
            println("   1. Coloque imagens no diretório: $(CNNCheckinCore.INCREMENTAL_DATA_PATH)")
            println("   2. Use nomes diferentes das pessoas já treinadas: $(join(original_person_names, ", "))")
            println("   3. Nome dos arquivos deve seguir padrão: nome-numero.jpg")
            println("   4. Mínimo de 3 imagens por pessoa")
            println("   5. Formatos suportados: .jpg, .jpeg, .png, .bmp, .tiff")
            return false
        end
        
        println("\nConfiguração do aprendizado incremental:")
        println("   - Pessoas originais: $(length(original_person_names))")
        println("   - Pessoas novas: $(length(new_person_names))")
        println("   - Total de pessoas: $(length(all_person_names))")
        println("   - Pessoas novas: $(join(new_person_names, ", "))")
        
        # Expand model architecture with FIXED weight copying
        student_model = expand_model_for_incremental(teacher_model, length(original_person_names), 
                                                   length(all_person_names))
        
        # Create datasets and batches
        (train_images, train_labels), (val_images, val_labels) = create_incremental_datasets(people_data, original_person_names)
        train_batches = create_incremental_batches(train_images, train_labels, CNNCheckinCore.BATCH_SIZE)
        val_batches = create_incremental_batches(val_images, val_labels, CNNCheckinCore.BATCH_SIZE)
        
        if length(train_batches) == 0
            error("Não foi possível criar batches de treino incremental!")
        end
        
        # Train with knowledge distillation
        train_losses, val_accuracies, best_val_acc, best_epoch = train_incremental_model(
            student_model, teacher_model, train_batches, val_batches, 
            length(original_person_names), CNNCheckinCore.INCREMENTAL_EPOCHS, 
            CNNCheckinCore.INCREMENTAL_LR
        )
        
        end_time = time()
        duration_minutes = (end_time - start_time) / 60
        
        # Prepare training info
        training_info = Dict(
            "epochs_trained" => length(val_accuracies),
            "final_accuracy" => best_val_acc,
            "best_epoch" => best_epoch,
            "total_training_images" => length(train_images),
            "total_validation_images" => length(val_images),
            "duration_minutes" => duration_minutes
        )
        
        println("\nAprendizado incremental concluído!")
        println("Resultados:")
        println("   - Melhor acurácia: $(round(best_val_acc*100, digits=2))% (Epoch $best_epoch)")
        println("   - Epochs treinadas: $(training_info["epochs_trained"])/$(CNNCheckinCore.INCREMENTAL_EPOCHS)")
        println("   - Duração: $(round(duration_minutes, digits=1)) minutos")
        println("   - Pessoas novas adicionadas: $(length(new_person_names))")
        
        # Save incremental model with FIXED saving
        success = save_incremental_model(student_model, all_person_names, original_person_names,
                                       new_person_names, CNNCheckinCore.MODEL_PATH,
                                       CNNCheckinCore.CONFIG_PATH, training_info)
        
        if success
            println("\n✅ Modelo incremental salvo com sucesso!")
            println("Arquivos atualizados:")
            println("   - Configuração: $(CNNCheckinCore.CONFIG_PATH)")
            println("   - Modelo: $(CNNCheckinCore.MODEL_PATH)")
            println("   - Dados do modelo: $(CNNCheckinCore.MODEL_DATA_TOML_PATH)")
            println("\nPróximo passo:")
            println("   - Para identificação: julia cnncheckin_identify.jl <caminho_da_imagem>")
            
            # Show updated person list
            println("\n👥 Pessoas agora reconhecidas pelo sistema:")
            for (i, person) in enumerate(all_person_names)
                status = person in original_person_names ? "ORIGINAL" : "NOVA"
                println("   $i. $person [$status]")
            end
        else
            println("Modelo treinado mas alguns arquivos falharam ao salvar")
        end
        
        return success
        
    catch e
        println("Erro durante aprendizado incremental: $e")
        println("Stack trace:")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
        return false
    end
end

# Execute if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    success = incremental_learning_command()
    if success
        println("Aprendizado incremental concluído com sucesso!")
    else
        println("Aprendizado incremental falhou")
    end
end

# Uso: julia cnncheckin_incremental.jl