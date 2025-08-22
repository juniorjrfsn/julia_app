using Flux
using Statistics
using Random
using JLD2
using Dates

include("cnncheckin_core.jl")
using .CNNCheckinCore

# Função para carregar dados das imagens (CORRIGIDA)
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
                println("✅ Carregada: $filename -> $person_name ($total_imgs variações)")
            else
                println("⚠️ Falha ao carregar: $filename")
            end
        end
    end
    
    people_data = Vector{CNNCheckinCore.PersonData}()
    person_names = sort(collect(keys(person_images)))
    
    # CORREÇÃO: Usar indexação começando em 1 (padrão Julia) e ordenar nomes
    for (idx, person_name) in enumerate(person_names)
        images = person_images[person_name]
        if length(images) > 0
            push!(people_data, CNNCheckinCore.PersonData(person_name, images, idx))
            println("👤 Pessoa: $person_name - $(length(images)) imagens (Label: $idx)")
        end
    end
    
    return people_data, person_names
end

# Função para criar datasets balanceados
function create_datasets(people_data::Vector{CNNCheckinCore.PersonData}, split_ratio::Float64 = 0.8)
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
        
        println("   - $((person.name)): $(n_train) treino, $(n_imgs - n_train) validação")
    end
    
    println("📈 Dataset criado:")
    println("   - Treino: $(length(train_images)) imagens")
    println("   - Validação: $(length(val_images)) imagens")
    
    return (train_images, train_labels), (val_images, val_labels)
end

# Função para criar batches (CORRIGIDA)
function create_batches(images, labels, batch_size)
    batches = []
    n_samples = length(images)
    if n_samples == 0
        return batches
    end
    
    # CORREÇÃO: Usar range correto para os labels
    unique_labels = unique(labels)
    min_label = minimum(unique_labels)
    max_label = maximum(unique_labels)
    label_range = min_label:max_label
    
    println("🔢 Labels únicos encontrados: $unique_labels")
    println("🔢 Range de labels para one-hot: $label_range")
    
    for i in 1:batch_size:n_samples
        end_idx = min(i + batch_size - 1, n_samples)
        batch_images = images[i:end_idx]
        batch_labels = labels[i:end_idx]
        batch_tensor = cat(batch_images..., dims=4)
        try
            batch_labels_onehot = Flux.onehotbatch(batch_labels, label_range)
            push!(batches, (batch_tensor, batch_labels_onehot))
            println("   ✅ Batch $(div(i-1, batch_size)+1): $(length(batch_labels)) amostras")
        catch e
            println("❌ Erro ao criar one-hot encoding para batch $i-$end_idx: $e")
            println("   Labels do batch: $batch_labels")
            println("   Range esperado: $label_range")
            continue
        end
    end
    
    return batches
end

# Arquitetura da CNN
function create_cnn_model(num_classes::Int, input_size::Tuple{Int, Int} = CNNCheckinCore.IMG_SIZE)
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

# Função para salvar modelo, configuração e dados do modelo (CORRIGIDA)
function save_model_and_config(model, person_names, model_filepath::String, config_filepath::String, training_info::Dict)
    println("💾 Salvando modelo, configuração e dados do modelo...")
    
    model_data = Dict(
        "model_state" => model,
        "person_names" => person_names,  # CORREÇÃO: Salvar nomes na ordem correta
        "timestamp" => string(Dates.now())
    )
    
    try
        jldsave(model_filepath; model_data=model_data)
        println("✅ Pesos do modelo salvos em: $model_filepath")
    catch e
        println("❌ Erro ao salvar modelo: $e")
        return false
    end
    
    model_data_saved = CNNCheckinCore.save_model_data_toml(model, person_names, CNNCheckinCore.MODEL_DATA_TOML_PATH)
    if !model_data_saved
        println("⚠️ Aviso: Falha ao salvar dados do modelo em TOML")
    end
    
    config = CNNCheckinCore.create_default_config()
    config["model"]["num_classes"] = length(person_names)
    config["model"]["augmentation_used"] = training_info["augmentation_used"]
    config["training"]["epochs_trained"] = training_info["epochs_trained"]
    config["training"]["final_accuracy"] = training_info["final_accuracy"]
    config["training"]["best_epoch"] = training_info["best_epoch"]
    config["data"]["person_names"] = person_names  # CORREÇÃO: Manter ordem correta
    config["data"]["timestamp"] = string(Dates.now())
    config["training_stats"] = Dict(
        "total_training_images" => training_info["total_training_images"],
        "total_validation_images" => training_info["total_validation_images"],
        "early_stopped" => training_info["early_stopped"],
        "training_duration_minutes" => get(training_info, "duration_minutes", 0.0)
    )
    config["files"] = Dict(
        "model_weights" => CNNCheckinCore.MODEL_PATH,
        "model_data_toml" => CNNCheckinCore.MODEL_DATA_TOML_PATH,
        "config_toml" => CNNCheckinCore.CONFIG_PATH
    )
    
    config_saved = CNNCheckinCore.save_config(config, config_filepath)
    
    # Verificar se os mapeamentos estão corretos
    println("🔍 Verificando mapeamentos:")
    for (i, name) in enumerate(person_names)
        println("   - Índice $i: $name")
    end
    
    return config_saved && model_data_saved
end

# Função de treinamento principal
function train_command()
    println("🎯 Sistema de Reconhecimento Facial - Modo Treinamento")
    
    start_time = time()
    
    try
        people_data, person_names = load_face_data(CNNCheckinCore.DATA_PATH; use_augmentation=true)
        if length(people_data) == 0
            error("Nenhum dado encontrado!")
        end
        
        num_classes = length(person_names)
        println("👥 Total de pessoas: $num_classes")
        total_images = sum(length(person.images) for person in people_data)
        println("📊 Total de imagens (com augmentation): $total_images")
        
        # Imprimir mapeamento pessoa->label para debug
        println("🏷️ Mapeamento pessoa->label:")
        for person in people_data
            println("   - $(person.name) -> Label $(person.label)")
        end
        
        (train_images, train_labels), (val_images, val_labels) = create_datasets(people_data)
        train_batches = create_batches(train_images, train_labels, CNNCheckinCore.BATCH_SIZE)
        val_batches = create_batches(val_images, val_labels, CNNCheckinCore.BATCH_SIZE)
        
        if length(train_batches) == 0
            error("Não foi possível criar batches de treinamento!")
        end
        
        println("🧠 Criando modelo CNN...")
        model = create_cnn_model(num_classes)
        train_losses, val_accuracies, best_val_acc, best_epoch = train_model(model, train_batches, val_batches, CNNCheckinCore.EPOCHS, CNNCheckinCore.LEARNING_RATE)
        
        end_time = time()
        duration_minutes = (end_time - start_time) / 60
        
        training_info = Dict(
            "epochs_trained" => length(val_accuracies),
            "final_accuracy" => best_val_acc,
            "best_epoch" => best_epoch,
            "total_training_images" => length(train_images),
            "total_validation_images" => length(val_images),
            "augmentation_used" => true,
            "early_stopped" => length(val_accuracies) < CNNCheckinCore.EPOCHS,
            "duration_minutes" => duration_minutes
        )
        
        println("🎉 Treinamento concluído!")
        println("📊 Resultados:")
        println("   - Melhor acurácia: $(round(best_val_acc*100, digits=2))% (Epoch $best_epoch)")
        println("   - Epochs treinados: $(training_info["epochs_trained"])/$(CNNCheckinCore.EPOCHS)")
        println("   - Duração: $(round(duration_minutes, digits=1)) minutos")
        println("   - Early stopping: $(training_info["early_stopped"] ? "Sim" : "Não")")
        
        success = save_model_and_config(model, person_names, CNNCheckinCore.MODEL_PATH, CNNCheckinCore.CONFIG_PATH, training_info)
        
        if success
            println("✅ Sistema treinado e salvo com sucesso!")
            println("📄 Arquivos gerados:")
            println("   - Configuração: $(CNNCheckinCore.CONFIG_PATH)")
            println("   - Pesos do modelo: $(CNNCheckinCore.MODEL_PATH)")
            println("   - Dados do modelo: $(CNNCheckinCore.MODEL_DATA_TOML_PATH)")
        else
            println("⚠️ Modelo treinado mas houve erro ao salvar alguns arquivos")
        end
        return success
    catch e
        println("❌ Erro durante treinamento: $e")
        return false
    end
end

# Executar comando se for chamado diretamente
if abspath(PROGRAM_FILE) == @__FILE__
    success = train_command()
    if success
        println("🎉 Treinamento concluído com sucesso!")
    else
        println("💥 Falha no treinamento")
    end
end



    # julia cnncheckin_train.jl
    # julia cnncheckin_identify.jl ../../../dados/fotos_teste/ela.jpg
    # julia cnncheckin_identify.jl ../../../dados/fotos_teste/teste.png
    # julia cnncheckin_identify.jl ../../../dados/fotos_teste/im_elo.jpg
    
    # julia cnncheckin_info.jl
    # julia cnncheckin_model_data.jl
    # julia cnncheckin_validate.jl
    # julia cnncheckin_export_config.jl backup_config_2025.toml
    # julia cnncheckin_help.jl


    # julia cnncheckin_debug.jl ../../../dados/fotos_teste/teste.png
    # julia cnncheckin_debug.jl --dir ../../../dados/fotos_teste/
