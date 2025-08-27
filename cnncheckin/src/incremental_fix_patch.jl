# Complete fix for incremental learning dimension issues
# File: incremental_fix_patch.jl
# Apply this to your cnncheckin_incremental.jl

using Flux
using Statistics
using Random
using JLD2
using Dates

include("cnncheckin_core.jl")
using .CNNCheckinCore

# FIXED: Create batches for incremental learning with proper dimensions
function create_incremental_batches_fixed(images, labels, batch_size, total_classes::Int)
    batches = []
    n_samples = length(images)
    if n_samples == 0
        return batches
    end
    
    unique_labels = unique(labels)
    # CRITICAL FIX: Use full class range (1 to total_classes)
    label_range = 1:total_classes
    
    println("Criando batches incrementais (CORRIGIDO):")
    println("   - Labels únicos presentes: $unique_labels")
    println("   - Range completo de classes: $label_range")
    println("   - Total de classes no modelo: $total_classes")
    
    for i in 1:batch_size:n_samples
        end_idx = min(i + batch_size - 1, n_samples)
        batch_images = images[i:end_idx]
        batch_labels = labels[i:end_idx]
        batch_tensor = cat(batch_images..., dims=4)
        
        try
            # FIXED: Use complete class range for proper one-hot encoding
            batch_labels_onehot = Flux.onehotbatch(batch_labels, label_range)
            
            # Verify dimensions
            expected_shape = (total_classes, length(batch_labels))
            actual_shape = size(batch_labels_onehot)
            
            if actual_shape == expected_shape
                push!(batches, (batch_tensor, batch_labels_onehot))
                println("   ✅ Batch $(div(i-1, batch_size)+1): $(length(batch_labels)) amostras - OneHot: $actual_shape")
            else
                println("   ❌ Batch $(div(i-1, batch_size)+1): Dimensões incorretas! Esperado: $expected_shape, Atual: $actual_shape")
                continue
            end
            
        catch e
            println("   ❌ Erro criando batch $i-$end_idx: $e")
            continue
        end
    end
    
    if length(batches) == 0
        error("Nenhum batch válido foi criado! Problema na codificação one-hot.")
    end
    
    println("   ✅ Total de batches criados: $(length(batches))")
    return batches
end

# FIXED: Create incremental datasets with proper class mapping
function create_incremental_datasets_fixed(people_data::Vector{CNNCheckinCore.PersonData}, 
                                          original_people::Vector{String}, split_ratio::Float64 = 0.8)
    println("Criando datasets para aprendizado incremental (CORRIGIDO)...")
    
    train_images = Vector{Array{Float32, 3}}()
    train_labels = Vector{Int}()
    val_images = Vector{Array{Float32, 3}}()
    val_labels = Vector{Int}()
    
    # Get all unique labels and ensure they're consecutive starting from 1
    all_labels = [p.label for p in people_data]
    total_classes = maximum(all_labels)  # Should match total number of people
    
    println("   - Labels presentes nos dados: $(sort(unique(all_labels)))")
    println("   - Total de classes esperado: $total_classes")
    
    # Separate original and new people
    original_data = filter(p -> p.name in original_people, people_data)
    new_data = filter(p -> !(p.name in original_people), people_data)
    
    println("   - Pessoas originais: $(length(original_data))")
    println("   - Pessoas novas: $(length(new_data))")
    
    # Process all people
    for person in people_data
        n_imgs = length(person.images)
        
        if person.is_incremental
            n_train = max(1, min(n_imgs - 1, Int(floor(n_imgs * 0.75))))
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
        println("   - $(person.name) [Label: $(person.label)] [$status]: $n_train treino, $(n_imgs - n_train) validação")
    end
    
    # Verify label consistency
    train_unique = sort(unique(train_labels))
    val_unique = sort(unique(val_labels))
    
    println("\nVerificação de labels:")
    println("   - Labels únicos treino: $train_unique")
    println("   - Labels únicos validação: $val_unique")
    println("   - Total esperado de classes: $total_classes")
    
    if maximum(vcat(train_labels, val_labels)) > total_classes
        error("Labels excedem número total de classes!")
    end
    
    return (train_images, train_labels), (val_images, val_labels), total_classes
end

# FIXED: Load incremental data with proper label assignment
function load_incremental_data_fixed(data_path::String, existing_people::Vector{String}; use_augmentation::Bool = true)
    println("Carregando dados incrementais (CORRIGIDO)...")
    
    if !isdir(data_path)
        error("Diretório não encontrado: $data_path")
    end
    
    person_images = Dict{String, Vector{Array{Float32, 3}}}()
    new_people = Set{String}()
    processed_files = 0
    failed_files = 0
    
    all_files = readdir(data_path)
    println("Encontrados $(length(all_files)) arquivos")
    
    for filename in all_files
        img_path = joinpath(data_path, filename)
        
        if !validate_incremental_image_file(img_path)
            failed_files += 1
            continue
        end
        
        try
            person_name = CNNCheckinCore.extract_person_name(filename)
            
            # Skip existing people
            if person_name in existing_people
                println("Pessoa já existe: $person_name - ignorando $filename")
                continue
            end
            
            push!(new_people, person_name)
            
            img_arrays = CNNCheckinCore.preprocess_image(img_path; augment=use_augmentation)
            
            if img_arrays !== nothing && length(img_arrays) > 0
                if !haskey(person_images, person_name)
                    person_images[person_name] = Vector{Array{Float32, 3}}()
                end
                
                for img_array in img_arrays
                    push!(person_images[person_name], img_array)
                end
                
                processed_files += 1
                println("✅ $filename -> $person_name ($(length(img_arrays)) variações)")
            end
            
        catch e
            println("❌ Erro processando $filename: $e")
            failed_files += 1
        end
    end
    
    println("\nResumo:")
    println("   - Processados: $processed_files")
    println("   - Falharam: $failed_files")
    println("   - Pessoas novas: $(length(new_people))")
    
    if length(new_people) == 0
        return Vector{CNNCheckinCore.PersonData}(), existing_people, String[]
    end
    
    # FIXED: Create proper label mapping
    all_person_names = vcat(existing_people, sort(collect(new_people)))
    people_data = Vector{CNNCheckinCore.PersonData}()
    
    println("\nMapeamento de labels (CORRIGIDO):")
    for (idx, person_name) in enumerate(all_person_names)
        if haskey(person_images, person_name)
            images = person_images[person_name]
            is_incremental = !(person_name in existing_people)
            
            # CRITICAL: Use 1-based consecutive indexing
            push!(people_data, CNNCheckinCore.PersonData(person_name, images, idx, is_incremental))
            status = is_incremental ? "NOVA" : "EXISTENTE"
            println("   $idx. $person_name [$status] - $(length(images)) imagens")
        end
    end
    
    # Filter people with insufficient data
    min_images = 3
    filtered_data = filter(p -> !p.is_incremental || length(p.images) >= min_images, people_data)
    filtered_new = [p.name for p in filtered_data if p.is_incremental]
    
    if length(filtered_data) < length(people_data)
        println("⚠️  Algumas pessoas foram removidas por dados insuficientes")
    end
    
    return filtered_data, all_person_names, filtered_new
end

# FIXED: Main incremental learning function
function incremental_learning_command_fixed()
    println("🧠 Sistema de Reconhecimento Facial - Modo Incremental (CORRIGIDO)")
    
    start_time = time()
    
    try
        # Load pre-trained model
        teacher_model, original_person_names, config = load_pretrained_model(
            CNNCheckinCore.MODEL_PATH, 
            CNNCheckinCore.CONFIG_PATH
        )
        
        # FIXED: Load incremental data with proper labeling
        people_data, all_person_names, new_person_names = load_incremental_data_fixed(
            CNNCheckinCore.INCREMENTAL_DATA_PATH, original_person_names; use_augmentation=true
        )
        
        if length(new_person_names) == 0
            println("\n❌ Nenhuma pessoa nova encontrada!")
            println("\n💡 Dicas para adicionar pessoas:")
            println("   1. Coloque imagens em: $(CNNCheckinCore.INCREMENTAL_DATA_PATH)")
            println("   2. Nomes de arquivo: nome-numero.jpg")
            println("   3. Pessoas existentes: $(join(original_person_names, ", "))")
            println("   4. Mínimo 3 imagens por pessoa nova")
            return false
        end
        
        println("\n📊 Configuração incremental:")
        println("   - Pessoas originais: $(length(original_person_names)) ($(join(original_person_names, ", ")))")
        println("   - Pessoas novas: $(length(new_person_names)) ($(join(new_person_names, ", ")))")
        println("   - Total final: $(length(all_person_names))")
        
        # Expand model
        student_model = expand_model_for_incremental(teacher_model, length(original_person_names), 
                                                   length(all_person_names))
        
        # FIXED: Create datasets with proper class handling
        (train_images, train_labels), (val_images, val_labels), total_classes = create_incremental_datasets_fixed(
            people_data, original_person_names)
        
        # FIXED: Create batches with correct dimensions
        train_batches = create_incremental_batches_fixed(train_images, train_labels, 
                                                       CNNCheckinCore.BATCH_SIZE, total_classes)
        val_batches = create_incremental_batches_fixed(val_images, val_labels, 
                                                     CNNCheckinCore.BATCH_SIZE, total_classes)
        
        if length(train_batches) == 0
            error("Falha ao criar batches de treino!")
        end
        
        println("\n🎯 Iniciando treinamento incremental...")
        
        # Train model
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
        
        println("\n🎉 Aprendizado incremental concluído!")
        println("📈 Resultados:")
        println("   - Melhor acurácia: $(round(best_val_acc*100, digits=2))% (Epoch $best_epoch)")
        println("   - Epochs: $(training_info["epochs_trained"])/$(CNNCheckinCore.INCREMENTAL_EPOCHS)")
        println("   - Duração: $(round(duration_minutes, digits=1)) minutos")
        
        # Save model
        success = save_incremental_model(student_model, all_person_names, original_person_names,
                                       new_person_names, CNNCheckinCore.MODEL_PATH,
                                       CNNCheckinCore.CONFIG_PATH, training_info)
        
        if success
            println("\n✅ Modelo salvo com sucesso!")
            println("\n👥 Sistema agora reconhece:")
            for (i, person) in enumerate(all_person_names)
                status = person in original_person_names ? "ORIGINAL" : "NOVA"
                println("   $i. $person [$status]")
            end
            
            println("\n🔍 Para testar:")
            println("   julia cnncheckin_identify.jl <caminho_da_imagem>")
        else
            println("❌ Erro salvando modelo")
        end
        
        return success
        
    catch e
        println("❌ Erro: $e")
        println("\nStack trace:")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
        return false
    end
end

# Test function to verify fix
function test_oneshot_encoding()
    println("🧪 Testando codificação one-hot...")
    
    # Test scenario: 3 classes total (junior=1, lele=2, cachorro=3)
    # But only cachorro (label=3) in incremental batch
    
    test_labels = [3, 3, 3, 3]  # Only cachorro
    total_classes = 3
    
    println("Labels de teste: $test_labels")
    println("Total de classes: $total_classes")
    
    try
        # Wrong way (what was happening before)
        wrong_range = minimum(test_labels):maximum(test_labels)  # 3:3
        wrong_onehot = Flux.onehotbatch(test_labels, wrong_range)
        println("❌ Forma incorreta - Range: $wrong_range, Shape: $(size(wrong_onehot))")
        
        # Right way (the fix)
        right_range = 1:total_classes  # 1:3
        right_onehot = Flux.onehotbatch(test_labels, right_range)
        println("✅ Forma correta - Range: $right_range, Shape: $(size(right_onehot))")
        
        println("\nOne-hot correto:")
        println(right_onehot)
        
    catch e
        println("Erro no teste: $e")
    end
end

# Run test
test_oneshot_encoding()

# Execute fixed incremental learning if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    success = incremental_learning_command_fixed()
    if success
        println("\n🎯 Correção aplicada com sucesso!")
    else
        println("\n❌ Ainda há problemas")
    end
end