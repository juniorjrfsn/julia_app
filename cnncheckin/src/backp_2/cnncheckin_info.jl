include("cnncheckin_core.jl")
using .CNNCheckinCore

# Função para exibir informações
function info_command()
    println("📋 Informações do Modelo de Reconhecimento Facial")
    
    if !isfile(CNNCheckinCore.CONFIG_PATH)
        println("❌ Configuração não encontrada. Execute primeiro: julia cnncheckin_train.jl")
        return false
    end
    
    try
        config = CNNCheckinCore.load_config(CNNCheckinCore.CONFIG_PATH)
        CNNCheckinCore.validate_config(config)
        model_data_toml = CNNCheckinCore.load_model_data_toml(CNNCheckinCore.MODEL_DATA_TOML_PATH)
        
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
                println("\n🔍 Últimas Predições ($(length(examples))):")
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
        
        println("\n🔧 Metadados:")
        println("   - Versão: $(config["metadata"]["version"])")
        println("   - Criado por: $(config["metadata"]["created_by"])")
        if haskey(config["metadata"], "last_saved")
            println("   - Última atualização: $(config["metadata"]["last_saved"])")
        end
        
        if haskey(config, "files")
            files = config["files"]
            println("\n📄 Arquivos:")
            for (key, filepath) in files
                status = isfile(filepath) ? "✅" : "❌"
                println("   - $(key): $(filepath) $status")
            end
        else
            println("\n📄 Arquivos:")
            println("   - Configuração: $(CNNCheckinCore.CONFIG_PATH) $(isfile(CNNCheckinCore.CONFIG_PATH) ? "✅" : "❌")")
            println("   - Modelo: $(CNNCheckinCore.MODEL_PATH) $(isfile(CNNCheckinCore.MODEL_PATH) ? "✅" : "❌")")
            println("   - Dados TOML: $(CNNCheckinCore.MODEL_DATA_TOML_PATH) $(isfile(CNNCheckinCore.MODEL_DATA_TOML_PATH) ? "✅" : "❌")")
        end
        return true
    catch e
        println("❌ Erro ao carregar informações: $e")
        return false
    end
end

# Executar comando se for chamado diretamente
if abspath(PROGRAM_FILE) == @__FILE__
    success = info_command()
    if !success
        println("💥 Falha ao obter informações")
    end
end