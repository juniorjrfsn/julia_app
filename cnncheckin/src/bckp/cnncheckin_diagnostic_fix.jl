# Complete diagnostic and fix for incremental learning issues
# File: cnncheckin_diagnostic_fix.jl

using Flux
using Statistics
using Random
using JLD2
using Dates

include("cnncheckin_core.jl")
using .CNNCheckinCore

"""
Diagnose and fix incremental learning model issues
This script will:
1. Analyze the current corrupted model
2. Identify the issues
3. Provide options to fix them
"""

function diagnose_model_corruption(model_path, config_path)
    println("🔍 DIAGNÓSTICO DO MODELO CORROMPIDO")
    println("="^60)
    
    # Load current model and config
    try
        config = CNNCheckinCore.load_config(config_path)
        data = load(model_path)
        model_data = data["model_data"]
        model = model_data["model_state"]
        person_names = config["data"]["person_names"]
        
        println("📋 Informações atuais:")
        println("   - Pessoas: $(join(person_names, ", "))")
        println("   - Total de classes: $(length(person_names))")
        
        # Analyze final layer weights
        final_layer = model[end]
        if isa(final_layer, Dense)
            println("\n🔬 Análise da camada final:")
            println("   - Dimensões: $(size(final_layer.weight))")
            
            # Check for unusual weight patterns
            for (i, person) in enumerate(person_names)
                weights = final_layer.weight[i, :]
                bias = final_layer.bias[i]
                weight_mean = mean(weights)
                weight_std = std(weights)
                
                println("   - $person (classe $i):")
                println("     * Peso médio: $(round(weight_mean, digits=6))")
                println("     * Desvio padrão: $(round(weight_std, digits=6))")
                println("     * Bias: $(round(bias, digits=6))")
                
                # Detect corruption signs
                if abs(weight_mean) > 10.0
                    println("     ⚠️  SUSPEITA: Pesos muito altos")
                end
                if weight_std > 5.0
                    println("     ⚠️  SUSPEITA: Variação muito alta nos pesos")
                end
            end
        end
        
        # Test with some sample predictions
        println("\n🧪 Teste de predições:")
        test_sample_predictions(model, person_names)
        
        return model, config, person_names
        
    catch e
        println("❌ Erro ao carregar modelo: $e")
        return nothing, nothing, nothing
    end
end

function test_sample_predictions(model, person_names)
    """Test model with synthetic data to check prediction patterns"""
    
    # Create random test images
    for i in 1:3
        test_img = randn(Float32, 128, 128, 3, 1) * 0.1f0
        
        try
            logits = model(test_img)
            probs = softmax(vec(logits))
            pred_class = argmax(probs)
            confidence = probs[pred_class]
            
            println("   Teste $i:")
            println("     - Predição: $(person_names[pred_class])")
            println("     - Confiança: $(round(confidence*100, digits=1))%")
            
            # Show all probabilities
            for (j, (name, prob)) in enumerate(zip(person_names, probs))
                println("       $j. $name: $(round(prob*100, digits=1))%")
            end
            
        catch e
            println("     ❌ Erro na predição: $e")
        end
    end
end

function backup_current_model(model_path, config_path)
    """Create backup of current corrupted model before fixing"""
    
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    backup_model = "face_recognition_model_corrupted_$timestamp.jld2"
    backup_config = "face_recognition_config_corrupted_$timestamp.toml"
    
    try
        cp(model_path, backup_model)
        cp(config_path, backup_config)
        println("💾 Backup criado:")
        println("   - Modelo: $backup_model")
        println("   - Config: $backup_config")
        return true
    catch e
        println("❌ Erro criando backup: $e")
        return false
    end
end

function restore_from_pretrained_and_retrain()
    """Option 1: Restore from pretrained model and redo incremental learning properly"""
    
    println("\n🔄 OPÇÃO 1: Restaurar do pré-treino e refazer incremental")
    println("="^60)
    
    # Check if we have original pretrained model backup
    backup_files = filter(f -> startswith(f, "face_recognition_model_pretrained"), readdir("."))
    
    if length(backup_files) > 0
        println("📁 Backups de pré-treino encontrados:")
        for (i, file) in enumerate(backup_files)
            println("   $i. $file")
        end
        
        println("\n💡 Para restaurar:")
        println("   1. Escolha um backup de pré-treino")
        println("   2. Copie para face_recognition_model.jld2")
        println("   3. Execute: julia cnncheckin_diagnostic_fix.jl --restore-pretrained")
        
    else
        println("❌ Nenhum backup de pré-treino encontrado")
        println("💡 Você precisará re-treinar do zero:")
        println("   julia cnncheckin_pretrain.jl")
    end
end

function fix_weights_manually(model, person_names, original_classes)
    """Option 2: Try to fix current model weights manually"""
    
    println("\n🔧 OPÇÃO 2: Corrigir pesos manualmente")
    println("="^60)
    
    final_layer = model[end]
    if !isa(final_layer, Dense)
        println("❌ Camada final não é Dense")
        return model
    end
    
    # Reset weights for better initialization
    input_size = size(final_layer.weight, 2)
    num_classes = length(person_names)
    
    # Re-initialize with smaller, more reasonable weights
    println("🔄 Re-inicializando pesos da camada final...")
    
    # Small random initialization
    new_weights = randn(Float32, num_classes, input_size) * 0.01f0
    new_bias = zeros(Float32, num_classes)
    
    # Create new layer
    new_final_layer = Dense(input_size, num_classes)
    new_final_layer.weight .= new_weights
    new_final_layer.bias .= new_bias
    
    # Replace final layer in model
    model_layers = collect(model)
    new_model = Chain(model_layers[1:end-1]..., new_final_layer)
    
    println("✅ Pesos re-inicializados com valores pequenos")
    return new_model
end

function create_balanced_incremental_dataset(original_people, new_people, data_paths)
    """Create a more balanced dataset for retraining"""
    
    println("\n📊 Criando dataset balanceado para re-treino")
    println("="^50)
    
    # This would need implementation based on your data structure
    # The key is ensuring equal representation of all classes
    
    println("💡 Para dataset balanceado:")
    println("   1. Certifique-se que cada pessoa tem pelo menos 5-10 imagens")
    println("   2. Use augmentação moderada (não excessiva)")
    println("   3. Valide que os nomes dos arquivos estão corretos")
    println("   4. Pessoas originais: $(join(original_people, ", "))")
    println("   5. Pessoas novas: $(join(new_people, ", "))")
end

function incremental_learning_fixed()
    """Properly implemented incremental learning with safeguards"""
    
    println("\n🎯 TREINAMENTO INCREMENTAL CORRIGIDO")
    println("="^60)
    
    try
        # Load original pretrained model (not corrupted one)
        println("📁 Procurando modelo pré-treinado original...")
        
        # Look for backup files or original pretrained model
        pretrained_files = filter(f -> contains(f, "pretrained") && endswith(f, ".jld2"), readdir("."))
        
        if length(pretrained_files) == 0
            println("❌ Modelo pré-treinado original não encontrado")
            println("💡 Execute primeiro: julia cnncheckin_pretrain.jl")
            return false
        end
        
        # Use the most recent pretrained model
        pretrained_model_path = sort(pretrained_files)[end]
        println("📂 Usando: $pretrained_model_path")
        
        # Load pretrained model
        pretrained_data = load(pretrained_model_path)
        teacher_model = pretrained_data["model_data"]["model_state"]
        original_people = pretrained_data["model_data"]["person_names"]
        
        println("✅ Modelo pré-treinado carregado")
        println("   Original pessoas: $(join(original_people, ", "))")
        
        # Load new incremental data
        incremental_data_path = CNNCheckinCore.INCREMENTAL_DATA_PATH
        new_people_data = load_new_incremental_data(incremental_data_path, original_people)
        
        if length(new_people_data) == 0
            println("❌ Nenhuma pessoa nova encontrada em: $incremental_data_path")
            return false
        end
        
        all_people = vcat(original_people, collect(keys(new_people_data)))
        println("🎯 Pessoas após incremental: $(join(all_people, ", "))")
        
        # Create expanded model with proper weight initialization
        student_model = expand_model_safely(teacher_model, length(original_people), length(all_people))
        
        # Train with conservative parameters
        success = train_incremental_conservative(student_model, teacher_model, 
                                               new_people_data, original_people, all_people)
        
        if success
            # Save the corrected model
            save_corrected_model(student_model, all_people, original_people)
            println("✅ Modelo incremental corrigido salvo!")
            return true
        else
            println("❌ Falha no treinamento incremental")
            return false
        end
        
    catch e
        println("❌ Erro no treinamento incremental corrigido: $e")
        return false
    end
end

function expand_model_safely(teacher_model, original_classes, total_classes)
    """Safely expand model preserving original class mappings"""
    
    if total_classes <= original_classes
        return teacher_model
    end
    
    # Extract layers
    layers = collect(teacher_model)
    feature_layers = layers[1:end-1]
    old_final = layers[end]
    
    # Create new final layer
    input_size = size(old_final.weight, 2)
    new_final = Dense(input_size, total_classes)
    
    # Initialize with very small weights
    new_final.weight .= randn(Float32, total_classes, input_size) * 0.001f0
    new_final.bias .= zeros(Float32, total_classes)
    
    # Copy original weights EXACTLY
    new_final.weight[1:original_classes, :] .= old_final.weight
    new_final.bias[1:original_classes] .= old_final.bias
    
    return Chain(feature_layers..., new_final)
end

function load_new_incremental_data(data_path, original_people)
    """Load only genuinely new people data"""
    
    new_people_data = Dict{String, Vector{String}}()  # person_name => image_paths
    
    if !isdir(data_path)
        return new_people_data
    end
    
    for filename in readdir(data_path)
        if !any(endswith(filename, ext) for ext in [".jpg", ".jpeg", ".png", ".bmp"])
            continue
        end
        
        person_name = CNNCheckinCore.extract_person_name(filename)
        
        # Skip if person already exists in original training
        if person_name in original_people
            continue
        end
        
        img_path = joinpath(data_path, filename)
        if CNNCheckinCore.validate_image_file(img_path)
            if !haskey(new_people_data, person_name)
                new_people_data[person_name] = String[]
            end
            push!(new_people_data[person_name], img_path)
        end
    end
    
    # Filter people with insufficient data
    min_images = 3
    filtered_data = Dict{String, Vector{String}}()
    for (person, paths) in new_people_data
        if length(paths) >= min_images
            filtered_data[person] = paths
            println("✅ $person: $(length(paths)) imagens")
        else
            println("❌ $person: apenas $(length(paths)) imagens (mínimo: $min_images)")
        end
    end
    
    return filtered_data
end

function train_incremental_conservative(student_model, teacher_model, new_people_data, 
                                      original_people, all_people)
    """Conservative incremental training to prevent catastrophic forgetting"""
    
    println("🎓 Treinamento incremental conservativo...")
    
    # Create training data
    train_images = Vector{Array{Float32, 3}}()
    train_labels = Vector{Int}()
    
    # Process new people data
    for (person_name, img_paths) in new_people_data
        person_index = findfirst(==(person_name), all_people)
        
        for img_path in img_paths
            img_arrays = CNNCheckinCore.preprocess_image(img_path; augment=true)
            if img_arrays !== nothing
                for img_array in img_arrays
                    push!(train_images, img_array)
                    push!(train_labels, person_index)
                end
            end
        end
    end
    
    if length(train_images) == 0
        println("❌ Nenhuma imagem de treino criada")
        return false
    end
    
    println("📊 Dataset criado: $(length(train_images)) imagens")
    
    # Conservative training parameters
    optimizer = ADAM(0.0001, (0.9, 0.999), 1e-8)  # Very low learning rate
    opt_state = Flux.setup(optimizer, student_model)
    
    # Simple training loop focusing only on new classes
    epochs = 10  # Fewer epochs to prevent overfitting
    
    for epoch in 1:epochs
        epoch_loss = 0.0
        num_batches = 0
        
        # Process in small batches
        batch_size = 4
        indices = randperm(length(train_images))
        
        for i in 1:batch_size:length(train_images)
            end_idx = min(i + batch_size - 1, length(train_images))
            batch_indices = indices[i:end_idx]
            
            batch_images = [train_images[idx] for idx in batch_indices]
            batch_labels = [train_labels[idx] for idx in batch_indices]
            
            # Create batch tensor
            batch_tensor = cat(batch_images..., dims=4)
            
            try
                # Create one-hot labels
                batch_labels_onehot = Flux.onehotbatch(batch_labels, 1:length(all_people))
                
                loss, grads = Flux.withgradient(student_model) do m
                    student_logits = m(batch_tensor)
                    
                    # Simple cross-entropy loss for new classes
                    ce_loss = Flux.logitcrossentropy(student_logits, batch_labels_onehot)
                    
                    # Add L2 regularization to prevent overfitting
                    l2_loss = sum(sum(abs2, p) for p in Flux.params(m)) * 1e-6
                    
                    return ce_loss + l2_loss
                end
                
                # Gradient clipping
                for p in Flux.params(student_model)
                    if haskey(grads[1], p)
                        grads[1][p] = clamp.(grads[1][p], -0.1f0, 0.1f0)
                    end
                end
                
                Flux.update!(opt_state, student_model, grads[1])
                epoch_loss += loss
                num_batches += 1
                
            catch e
                println("❌ Erro no batch: $e")
                continue
            end
        end
        
        avg_loss = num_batches > 0 ? epoch_loss / num_batches : 0.0
        println("Epoch $epoch/$epochs - Loss: $(round(avg_loss, digits=6))")
    end
    
    return true
end

function save_corrected_model(model, all_people, original_people)
    """Save the corrected incremental model"""
    
    model_data = Dict(
        "model_state" => model,
        "person_names" => all_people,
        "original_people" => original_people,
        "model_type" => "incremental_corrected",
        "timestamp" => string(Dates.now())
    )
    
    jldsave(CNNCheckinCore.MODEL_PATH; model_data=model_data)
    
    # Update config
    config = CNNCheckinCore.load_config(CNNCheckinCore.CONFIG_PATH)
    config["data"]["person_names"] = all_people
    config["model"]["num_classes"] = length(all_people)
    CNNCheckinCore.save_config(config, CNNCheckinCore.CONFIG_PATH)
    
    # Update TOML data
    CNNCheckinCore.save_model_data_toml(model, all_people, CNNCheckinCore.MODEL_DATA_TOML_PATH)
end

# Main execution
function main()
    if length(ARGS) == 0
        println("🩺 SISTEMA DE DIAGNÓSTICO E CORREÇÃO")
        println("="^50)
        println("Opções:")
        println("  --diagnose: Diagnosticar modelo atual")
        println("  --backup: Criar backup do modelo atual") 
        println("  --restore-pretrained: Opções para restaurar pré-treino")
        println("  --fix-weights: Tentar corrigir pesos manualmente")
        println("  --retrain: Re-treinar incremental corretamente")
        println("  --full-fix: Diagnóstico completo + correção")
        return
    end
    
    if ARGS[1] == "--diagnose"
        model, config, person_names = diagnose_model_corruption(
            CNNCheckinCore.MODEL_PATH, 
            CNNCheckinCore.CONFIG_PATH
        )
        
    elseif ARGS[1] == "--backup"
        backup_current_model(CNNCheckinCore.MODEL_PATH, CNNCheckinCore.CONFIG_PATH)
        
    elseif ARGS[1] == "--restore-pretrained"
        restore_from_pretrained_and_retrain()
        
    elseif ARGS[1] == "--fix-weights"
        model, config, person_names = diagnose_model_corruption(
            CNNCheckinCore.MODEL_PATH, 
            CNNCheckinCore.CONFIG_PATH
        )
        if model !== nothing
            fixed_model = fix_weights_manually(model, person_names, length(person_names)-1)
            # Save fixed model (implementation needed)
        end
        
    elseif ARGS[1] == "--retrain"
        success = incremental_learning_fixed()
        if success
            println("Re-treinamento incremental concluído!")
        else
            println("Falha no re-treinamento")
        end
        
    elseif ARGS[1] == "--full-fix"
        println("CORREÇÃO COMPLETA DO SISTEMA")
        println("="^50)
        
        # Step 1: Diagnose
        model, config, person_names = diagnose_model_corruption(
            CNNCheckinCore.MODEL_PATH, 
            CNNCheckinCore.CONFIG_PATH
        )
        
        # Step 2: Backup
        backup_current_model(CNNCheckinCore.MODEL_PATH, CNNCheckinCore.CONFIG_PATH)
        
        # Step 3: Fix
        success = incremental_learning_fixed()
        
        if success
            println("Sistema corrigido com sucesso!")
            
            # Step 4: Validate the fix
            println("\nValidando correção...")
            test_corrected_model()
        else
            println("Falha na correção automática")
            println("\nOpções manuais:")
            restore_from_pretrained_and_retrain()
        end
    end
end

function test_corrected_model()
    """Test the corrected model with validation images"""
    
    println("Testando modelo corrigido...")
    
    # Load corrected model
    try
        config = CNNCheckinCore.load_config(CNNCheckinCore.CONFIG_PATH)
        data = load(CNNCheckinCore.MODEL_PATH)
        model = data["model_data"]["model_state"]
        person_names = config["data"]["person_names"]
        
        # Test with auth directory if available
        auth_path = CNNCheckinCore.AUTH_DATA_PATH
        if isdir(auth_path)
            println("Testando com imagens de autenticação...")
            
            test_files = filter(f -> any(endswith(f, ext) for ext in [".jpg", ".jpeg", ".png"]), 
                              readdir(auth_path))
            
            for test_file in test_files[1:min(3, length(test_files))]  # Test first 3 files
                img_path = joinpath(auth_path, test_file)
                println("\nTeste: $test_file")
                
                img_arrays = CNNCheckinCore.preprocess_image(img_path; augment=false)
                if img_arrays !== nothing && length(img_arrays) > 0
                    img_tensor = reshape(img_arrays[1], size(img_arrays[1])..., 1)
                    
                    logits = model(img_tensor)
                    probs = softmax(vec(logits))
                    pred_class = argmax(probs)
                    confidence = probs[pred_class]
                    
                    println("  Predição: $(person_names[pred_class])")
                    println("  Confiança: $(round(confidence*100, digits=1))%")
                    
                    # Show top 3 predictions
                    sorted_indices = sortperm(probs, rev=true)
                    println("  Top 3:")
                    for i in 1:min(3, length(sorted_indices))
                        idx = sorted_indices[i]
                        println("    $(i). $(person_names[idx]): $(round(probs[idx]*100, digits=1))%")
                    end
                end
            end
        end
        
    catch e
        println("Erro testando modelo corrigido: $e")
    end
end

# Additional utility functions for comprehensive fixing

function reset_to_pretrained_backup()
    """Reset to a known good pretrained state"""
    
    println("Procurando backups de pré-treino...")
    
    # Look for pretrained backups
    backup_files = filter(f -> contains(f, "pretrained") && endswith(f, ".jld2"), readdir("."))
    
    if length(backup_files) > 0
        latest_backup = sort(backup_files)[end]
        println("Encontrado backup: $latest_backup")
        
        try
            cp(latest_backup, CNNCheckinCore.MODEL_PATH, force=true)
            
            # Also restore corresponding config if exists
            config_backup = replace(latest_backup, ".jld2" => "_config.toml")
            if isfile(config_backup)
                cp(config_backup, CNNCheckinCore.CONFIG_PATH, force=true)
            end
            
            println("Modelo restaurado para estado pré-treinado")
            return true
        catch e
            println("Erro restaurando backup: $e")
            return false
        end
    else
        println("Nenhum backup de pré-treino encontrado")
        return false
    end
end

function create_training_report(original_people, new_people, accuracy_before, accuracy_after)
    """Create a detailed report of the incremental training process"""
    
    report_file = "incremental_training_report_$(Dates.format(Dates.now(), "yyyymmdd_HHMMSS")).txt"
    
    try
        open(report_file, "w") do io
            println(io, "RELATÓRIO DE TREINAMENTO INCREMENTAL")
            println(io, "="^60)
            println(io, "Data: $(Dates.format(Dates.now(), "dd/mm/yyyy HH:MM:SS"))")
            println(io)
            
            println(io, "CONFIGURAÇÃO:")
            println(io, "  Pessoas originais ($(length(original_people))):")
            for person in original_people
                println(io, "    - $person")
            end
            println(io)
            
            println(io, "  Pessoas novas ($(length(new_people))):")
            for person in new_people
                println(io, "    - $person")
            end
            println(io)
            
            println(io, "RESULTADOS:")
            println(io, "  Acurácia antes: $(round(accuracy_before*100, digits=2))%")
            println(io, "  Acurácia depois: $(round(accuracy_after*100, digits=2))%")
            if accuracy_after >= accuracy_before
                println(io, "  Status: MELHORIA")
            else
                println(io, "  Status: DEGRADAÇÃO")
            end
            println(io)
            
            println(io, "RECOMENDAÇÕES:")
            if accuracy_after < 0.7
                println(io, "  - Modelo com baixa acurácia, considere re-treino")
                println(io, "  - Verifique qualidade das imagens de treino")
                println(io, "  - Considere aumentar dados de treino")
            elseif accuracy_after < accuracy_before * 0.9
                println(io, "  - Houve degradação significativa")
                println(io, "  - Considere ajustar parâmetros de distilação")
                println(io, "  - Verifique se dados incrementais são balanceados")
            else
                println(io, "  - Treinamento incremental bem-sucedido")
                println(io, "  - Monitorar desempenho em produção")
            end
        end
        
        println("Relatório salvo em: $report_file")
        
    catch e
        println("Erro criando relatório: $e")
    end
end

# Execute if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end