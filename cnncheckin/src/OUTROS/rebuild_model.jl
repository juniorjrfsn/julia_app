# Rebuild Model Script - Fix corrupted or mismatched models
# File: rebuild_model.jl

using Flux
using JLD2
using TOML
using Dates

include("cnncheckin_core.jl")
using .CNNCheckinCore

println("🔧 RECONSTRUÇÃO DO MODELO")
println("="^70)

function create_backup()
    timestamp = replace(string(now()), ":" => "-", "." => "-")
    backup_name = "face_recognition_model_backup_$timestamp.jld2"
    
    try
        if isfile("face_recognition_model.jld2")
            cp("face_recognition_model.jld2", backup_name)
            println("✅ Backup criado: $backup_name")
            return true
        else
            println("⚠️  Nenhum modelo para fazer backup")
            return false
        end
    catch e
        println("❌ Erro criando backup: $e")
        return false
    end
end

function extract_feature_layers(old_model)
    """Extract feature extraction layers (all except final Dense)"""
    layers = collect(old_model)
    
    # Find last Dense layer
    last_dense_idx = 0
    for i in length(layers):-1:1
        if isa(layers[i], Dense)
            last_dense_idx = i
            break
        end
    end
    
    if last_dense_idx == 0
        error("Nenhuma camada Dense encontrada no modelo")
    end
    
    # Return all layers before the last Dense
    return layers[1:last_dense_idx-1], last_dense_idx
end

function rebuild_model_architecture(num_classes::Int)
    """Rebuild complete model architecture from scratch"""
    println("   Construindo arquitetura para $num_classes classes...")
    
    final_size = div(div(div(div(CNNCheckinCore.IMG_SIZE[1], 2), 2), 2), 2)
    final_features = 256 * final_size * final_size
    
    model = Chain(
        # Feature extraction
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
        
        # Classifier
        Flux.flatten,
        Dense(final_features, 512, relu),
        Dropout(0.4),
        Dense(512, 256, relu),
        Dropout(0.3),
        Dense(256, num_classes)
    )
    
    println("   ✅ Arquitetura criada")
    return model
end

function copy_weights_carefully(new_model, old_model)
    """Copy weights from old model to new model layer by layer"""
    println("   Copiando pesos do modelo antigo...")
    
    new_layers = collect(new_model)
    old_layers = collect(old_model)
    
    copied_count = 0
    skipped_count = 0
    
    # Copy layer by layer, matching by type and size
    for (i, new_layer) in enumerate(new_layers)
        if i > length(old_layers)
            println("      Camada $i: nova camada, pesos aleatórios")
            skipped_count += 1
            continue
        end
        
        old_layer = old_layers[i]
        
        # Check if layers are compatible
        if typeof(new_layer) != typeof(old_layer)
            println("      Camada $i: tipo diferente ($(typeof(old_layer)) → $(typeof(new_layer))), pulando")
            skipped_count += 1
            continue
        end
        
        # Try to copy weights
        try
            if isa(new_layer, Conv) || isa(new_layer, Dense)
                if hasfield(typeof(new_layer), :weight) && hasfield(typeof(old_layer), :weight)
                    if size(new_layer.weight) == size(old_layer.weight)
                        new_layer.weight .= old_layer.weight
                        new_layer.bias .= old_layer.bias
                        copied_count += 1
                        println("      Camada $i ($(typeof(new_layer))): ✅ copiado")
                    else
                        println("      Camada $i: tamanhos diferentes, reinicializando")
                        skipped_count += 1
                    end
                end
            elseif isa(new_layer, BatchNorm)
                if hasfield(typeof(new_layer), :β)
                    if length(new_layer.β) == length(old_layer.β)
                        new_layer.β .= old_layer.β
                        new_layer.γ .= old_layer.γ
                        copied_count += 1
                        println("      Camada $i (BatchNorm): ✅ copiado")
                    end
                end
            end
        catch e
            println("      Camada $i: erro ao copiar ($e), pulando")
            skipped_count += 1
        end
    end
    
    println("\n   📊 Resumo da cópia:")
    println("      - Camadas copiadas: $copied_count")
    println("      - Camadas puladas: $skipped_count")
    println("      - Total de camadas: $(length(new_layers))")
    
    return copied_count > 0
end

function rebuild_with_feature_preservation()
    println("\n🔨 OPÇÃO 1: Reconstruir preservando features treinadas")
    println("   (Mantém camadas convolucionais, reconstrói classificador)")
    
    try
        # Load old model and config
        data = load("face_recognition_model.jld2")
        old_model = data["model_data"]["model_state"]
        
        config = TOML.parsefile("face_recognition_config.toml")
        person_names = config["data"]["person_names"]
        num_classes = length(person_names)
        
        println("\n   Pessoas no sistema: $(join(person_names, ", "))")
        println("   Total de classes: $num_classes")
        
        # Build new architecture
        new_model = rebuild_model_architecture(num_classes)
        
        # Try to copy weights
        success = copy_weights_carefully(new_model, old_model)
        
        if !success
            println("   ⚠️  Falha ao copiar pesos, modelo será reinicializado")
        end
        
        # Test the new model
        println("\n   🧪 Testando novo modelo...")
        test_input = randn(Float32, 128, 128, 3, 1)
        test_output = new_model(test_input)
        
        if size(test_output, 1) == num_classes
            println("   ✅ Modelo funciona corretamente!")
            println("      Output shape: $(size(test_output))")
            
            # Save the rebuilt model
            model_data = Dict(
                "model_state" => new_model,
                "person_names" => person_names,
                "model_type" => "rebuilt",
                "timestamp" => string(now()),
                "rebuild_reason" => "Dimension mismatch or corruption"
            )
            
            jldsave("face_recognition_model.jld2"; model_data=model_data)
            
            # Update config
            config["model"]["num_classes"] = num_classes
            config["metadata"]["last_rebuild"] = string(now())
            CNNCheckinCore.save_config(config, "face_recognition_config.toml")
            
            println("\n   ✅ Modelo reconstruído e salvo!")
            println("\n   ⚠️  IMPORTANTE: Você precisa re-treinar o modelo:")
            
            if success
                println("      - Treino incremental (mais rápido): julia cnncheckin_incremental.jl")
            else
                println("      - Treino completo: julia cnncheckin_pretrain.jl")
            end
            
            return true
        else
            println("   ❌ Erro: output shape incorreto!")
            return false
        end
        
    catch e
        println("   ❌ Erro durante reconstrução: $e")
        println("\n   Stack trace:")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
        return false
    end
end

function rebuild_from_scratch()
    println("\n🔨 OPÇÃO 2: Reconstruir do zero")
    println("   (Remove modelo atual, cria novo não treinado)")
    
    try
        config = TOML.parsefile("face_recognition_config.toml")
        person_names = config["data"]["person_names"]
        num_classes = length(person_names)
        
        println("\n   Criando modelo novo para $num_classes classes...")
        
        # Build completely new model
        new_model = rebuild_model_architecture(num_classes)
        
        # Test it
        test_input = randn(Float32, 128, 128, 3, 1)
        test_output = new_model(test_input)
        
        println("   ✅ Modelo criado com sucesso!")
        println("      Output shape: $(size(test_output))")
        
        # Save
        model_data = Dict(
            "model_state" => new_model,
            "person_names" => person_names,
            "model_type" => "fresh",
            "timestamp" => string(now()),
            "training_info" => Dict(
                "epochs_trained" => 0,
                "final_accuracy" => 0.0
            )
        )
        
        jldsave("face_recognition_model.jld2"; model_data=model_data)
        
        # Update config
        config["training"]["epochs_trained"] = 0
        config["training"]["final_accuracy"] = 0.0
        config["metadata"]["last_rebuild"] = string(now())
        CNNCheckinCore.save_config(config, "face_recognition_config.toml")
        
        println("\n   ✅ Modelo não treinado salvo!")
        println("\n   🚀 PRÓXIMO PASSO OBRIGATÓRIO:")
        println("      julia cnncheckin_pretrain.jl")
        
        return true
        
    catch e
        println("   ❌ Erro: $e")
        return false
    end
end

function main()
    println("\nEste script reconstrói o modelo neural quando há problemas de dimensão")
    println("ou corrupção nos pesos.\n")
    
    # Create backup first
    println("📦 Criando backup do modelo atual...")
    create_backup()
    
    println("\n" * "="^70)
    println("ESCOLHA UMA OPÇÃO:")
    println("="^70)
    println("1. Reconstruir preservando features (recomendado se modelo foi treinado)")
    println("   - Tenta manter camadas convolucionais aprendidas")
    println("   - Reconstrói apenas classificador")
    println("   - Requer re-treino incremental (mais rápido)")
    println()
    println("2. Reconstruir do zero")
    println("   - Cria modelo completamente novo")
    println("   - Remove todo treinamento anterior")
    println("   - Requer re-treino completo")
    println()
    println("3. Cancelar")
    
    print("\nEscolha (1-3): ")
    choice = readline()
    
    if choice == "1"
        success = rebuild_with_feature_preservation()
    elseif choice == "2"
        println("\n⚠️  ATENÇÃO: Isto vai remover TODO o treinamento anterior!")
        print("Tem certeza? (sim/não): ")
        confirm = lowercase(readline())
        
        if confirm == "sim" || confirm == "s"
            success = rebuild_from_scratch()
        else
            println("Operação cancelada")
            return
        end
    else
        println("Operação cancelada")
        return
    end
    
    if success
        println("\n" * "="^70)
        println("✅ RECONSTRUÇÃO CONCLUÍDA COM SUCESSO!")
        println("="^70)
    else
        println("\n" * "="^70)
        println("❌ RECONSTRUÇÃO FALHOU")
        println("="^70)
        println("\nTente a opção 2 (reconstruir do zero) ou")
        println("execute: julia cnncheckin_pretrain.jl")
    end
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end