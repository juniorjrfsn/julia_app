# projeto: webcamcnn
# file: webcamcnn/src/weights_utils.jl


using TOML
using Dates
using Statistics

include("weights_manager.jl")

# Função para limpar treinamentos antigos
function cleanup_old_trainings(filepath::String, keep_last_n::Int = 5)
    println("🧹 Limpando treinamentos antigos (mantendo os últimos $keep_last_n)...")
    
    if !isfile(filepath)
        println("❌ Arquivo não encontrado: $filepath")
        return false
    end
    
    try
        data = TOML.parsefile(filepath)
        
        if !haskey(data, "trainings") || length(data["trainings"]) <= keep_last_n
            println("ℹ️ Não há treinamentos suficientes para limpeza")
            return true
        end
        
        # Ordenar treinamentos por timestamp
        trainings = collect(data["trainings"])
        sort!(trainings, by=x -> x[2]["metadata"]["timestamp"])
        
        # Manter apenas os últimos N
        trainings_to_keep = trainings[end-keep_last_n+1:end]
        
        # Reconstruir estrutura de dados
        new_trainings = Dict{String, Any}()
        for (training_id, training_data) in trainings_to_keep
            new_trainings[training_id] = training_data
        end
        
        data["trainings"] = new_trainings
        data["summary"]["total_trainings"] = length(new_trainings)
        data["summary"]["latest_training"] = trainings_to_keep[end][1]
        data["format_info"]["last_updated"] = string(Dates.now())
        
        # Salvar arquivo atualizado
        open(filepath, "w") do io
            TOML.print(io, data)
        end
        
        removed_count = length(trainings) - keep_last_n
        println("✅ Removidos $removed_count treinamentos antigos")
        println("📊 Treinamentos restantes: $(length(new_trainings))")
        
        return true
        
    catch e
        println("❌ Erro durante limpeza: $e")
        return false
    end
end

# Função para exportar um treinamento específico
function export_training(filepath::String, training_id::String, export_path::String)
    println("📤 Exportando treinamento: $training_id")
    
    training_data = load_weights_from_toml(filepath, training_id)
    if training_data === nothing
        return false
    end
    
    # Criar estrutura exportada
    exported_data = Dict{String, Any}(
        "exported_training" => Dict(training_id => training_data),
        "export_info" => Dict(
            "exported_at" => string(Dates.now()),
            "original_file" => filepath,
            "training_id" => training_id,
            "export_version" => "1.0"
        )
    )
    
    try
        open(export_path, "w") do io
            TOML.print(io, exported_data)
        end
        println("✅ Treinamento exportado para: $export_path")
        return true
    catch e
        println("❌ Erro na exportação: $e")
        return false
    end
end

# Função para importar um treinamento de arquivo externo
function import_training(source_path::String, target_path::String)
    println("📥 Importando treinamento de: $source_path")
    
    try
        exported_data = TOML.parsefile(source_path)
        
        if !haskey(exported_data, "exported_training")
            println("❌ Arquivo não contém treinamento válido para importação")
            return false
        end
        
        # Carregar dados de destino existentes
        target_data = Dict{String, Any}()
        if isfile(target_path)
            target_data = TOML.parsefile(target_path)
        else
            target_data = Dict(
                "trainings" => Dict{String, Any}(),
                "format_info" => Dict(
                    "version" => "1.0",
                    "description" => "CNN Face Recognition Weights and Biases",
                    "created_by" => "webcamcnn.jl"
                )
            )
        end
        
        if !haskey(target_data, "trainings")
            target_data["trainings"] = Dict{String, Any}()
        end
        
        # Adicionar treinamento importado
        for (training_id, training_data) in exported_data["exported_training"]
            if haskey(target_data["trainings"], training_id)
                new_id = "$(training_id)_imported_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
                println("⚠️ ID já existe, renomeando para: $new_id")
                target_data["trainings"][new_id] = training_data
            else
                target_data["trainings"][training_id] = training_data
            end
        end
        
        # Atualizar metadados
        target_data["summary"] = Dict(
            "total_trainings" => length(target_data["trainings"]),
            "latest_training" => collect(keys(target_data["trainings"]))[end],
            "all_persons" => collect(Set(vcat([t["metadata"]["person_names"] 
                                             for t in values(target_data["trainings"])]...)))
        )
        target_data["format_info"]["last_updated"] = string(Dates.now())
        
        # Salvar arquivo atualizado
        open(target_path, "w") do io
            TOML.print(io, target_data)
        end
        
        println("✅ Treinamento importado com sucesso")
        return true
        
    catch e
        println("❌ Erro na importação: $e")
        return false
    end
end

# Função para calcular estatísticas de evolução dos treinamentos
function analyze_training_evolution(filepath::String)
    println("📈 Analisando evolução dos treinamentos...")
    
    if !isfile(filepath)
        println("❌ Arquivo não encontrado: $filepath")
        return false
    end
    
    try
        data = TOML.parsefile(filepath)
        trainings = data["trainings"]
        
        if length(trainings) < 2
            println("ℹ️ São necessários pelo menos 2 treinamentos para análise")
            return false
        end
        
        # Coletar dados dos treinamentos ordenados por timestamp
        training_list = collect(trainings)
        sort!(training_list, by=x -> x[2]["metadata"]["timestamp"])
        
        timestamps = []
        accuracies = []
        epochs_list = []
        param_counts = []
        person_counts = []
        
        for (training_id, training_data) in training_list
            meta = training_data["metadata"]
            stats = training_data["model_stats"]
            
            push!(timestamps, meta["timestamp"])
            push!(accuracies, meta["final_accuracy"])
            push!(epochs_list, meta["epochs_trained"])
            push!(param_counts, stats["total_parameters"])
            push!(person_counts, length(meta["person_names"]))
        end
        
        println("\n📊 ANÁLISE DE EVOLUÇÃO DOS TREINAMENTOS")
        println("=" * 50)
        
        # Estatísticas de acurácia
        acc_improvement = accuracies[end] - accuracies[1]
        println("🎯 Evolução da Acurácia:")
        println("   Primeira: $(round(accuracies[1]*100, digits=2))%")
        println("   Última: $(round(accuracies[end]*100, digits=2))%")
        println("   Melhoria: $(acc_improvement > 0 ? "+" : "")$(round(acc_improvement*100, digits=2))%")
        println("   Melhor: $(round(maximum(accuracies)*100, digits=2))%")
        println("   Média: $(round(mean(accuracies)*100, digits=2))%")
        
        # Estatísticas de épocas
        println("\n⏱️  Evolução das Épocas:")
        println("   Média de épocas: $(round(mean(epochs_list), digits=1))")
        println("   Variação: $(minimum(epochs_list)) - $(maximum(epochs_list))")
        
        # Estatísticas de pessoas
        println("\n👥 Evolução do Dataset:")
        println("   Pessoas inicial: $(person_counts[1])")
        println("   Pessoas atual: $(person_counts[end])")
        println("   Crescimento: $(person_counts[end] - person_counts[1]) pessoas")
        
        # Estatísticas de parâmetros
        println("\n🧮 Parâmetros do Modelo:")
        println("   Parâmetros: $(param_counts[end])")
        println("   Tamanho: $(round(param_counts[end] * 4 / (1024^2), digits=2)) MB")
        
        # Tendências
        println("\n📈 Tendências:")
        if length(accuracies) >= 3
            recent_trend = mean(accuracies[end-2:end]) - mean(accuracies[1:3])
            trend_desc = recent_trend > 0.01 ? "Melhorando" : recent_trend < -0.01 ? "Piorando" : "Estável"
            println("   Acurácia: $trend_desc ($(round(recent_trend*100, digits=2))%)")
        end
        
        if std(epochs_list) > 2
            println("   Convergência: Variável")
        else
            println("   Convergência: Consistente")
        end
        
        return true
        
    catch e
        println("❌ Erro na análise: $e")
        return false
    end
end

# Função para criar backup dos pesos
function backup_weights(filepath::String, backup_dir::String = "backups")
    println("💾 Criando backup dos pesos...")
    
    if !isfile(filepath)
        println("❌ Arquivo não encontrado: $filepath")
        return false
    end
    
    try
        if !isdir(backup_dir)
            mkpath(backup_dir)
        end
        
        timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
        backup_filename = "weights_backup_$timestamp.toml"
        backup_path = joinpath(backup_dir, backup_filename)
        
        cp(filepath, backup_path)
        
        println("✅ Backup criado: $backup_path")
        return true
        
    catch e
        println("❌ Erro no backup: $e")
        return false
    end
end

# Função para validar integridade do arquivo de pesos
function validate_weights_file(filepath::String)
    println("🔍 Validando integridade do arquivo de pesos...")
    
    if !isfile(filepath)
        println("❌ Arquivo não encontrado: $filepath")
        return false
    end
    
    try
        data = TOML.parsefile(filepath)
        issues = []
        
        # Verificar estrutura básica
        if !haskey(data, "trainings")
            push!(issues, "Seção 'trainings' não encontrada")
        end
        
        if !haskey(data, "format_info")
            push!(issues, "Seção 'format_info' não encontrada")
        end
        
        # Verificar cada treinamento
        if haskey(data, "trainings")
            for (training_id, training_data) in data["trainings"]
                if !haskey(training_data, "metadata")
                    push!(issues, "Training '$training_id': metadados ausentes")
                end
                
                if !haskey(training_data, "layers")
                    push!(issues, "Training '$training_id': camadas ausentes")
                end
                
                if haskey(training_data, "metadata")
                    required_fields = ["timestamp", "person_names", "final_accuracy"]
                    for field in required_fields
                        if !haskey(training_data["metadata"], field)
                            push!(issues, "Training '$training_id': campo '$field' ausente nos metadados")
                        end
                    end
                end
            end
        end
        
        if isempty(issues)
            println("✅ Arquivo válido e íntegro")
            return true
        else
            println("⚠️ Problemas encontrados:")
            for issue in issues
                println("   - $issue")
            end
            return false
        end
        
    catch e
        println("❌ Erro ao validar arquivo: $e")
        return false
    end
end

# Menu principal para utilitários de peso
function weights_utilities_menu()
    while true
        println("\n🛠️  UTILITÁRIOS DE GERENCIAMENTO DE PESOS")
        println("=" * 50)
        println("1 - Listar treinamentos")
        println("2 - Analisar evolução dos treinamentos")
        println("3 - Limpar treinamentos antigos")
        println("4 - Exportar treinamento específico")
        println("5 - Importar treinamento")
        println("6 - Criar backup")
        println("7 - Validar integridade do arquivo")
        println("8 - Voltar")
        
        print("Escolha uma opção: ")
        choice = strip(readline())
        
        weights_file = "model_weights.toml"
        
        if choice == "1"
            list_saved_trainings(weights_file)
        elseif choice == "2"
            analyze_training_evolution(weights_file)
        elseif choice == "3"
            print("Quantos treinamentos manter? (padrão: 5): ")
            keep_n = readline()
            keep_n = isempty(keep_n) ? 5 : parse(Int, keep_n)
            cleanup_old_trainings(weights_file, keep_n)
        elseif choice == "4"
            list_saved_trainings(weights_file)
            print("ID do treinamento para exportar: ")
            training_id = strip(readline())
            if !isempty(training_id)
                export_path = "exported_$training_id.toml"
                export_training(weights_file, training_id, export_path)
            end
        elseif choice == "5"
            print("Caminho do arquivo para importar: ")
            source_path = strip(readline())
            if !isempty(source_path)
                import_training(source_path, weights_file)
            end
        elseif choice == "6"
            backup_weights(weights_file)
        elseif choice == "7"
            validate_weights_file(weights_file)
        elseif choice == "8"
            return false
        else
            println("❌ Opção inválida!")
        end
        
        println("\nPressione Enter para continuar...")
        readline()
    end
end

# Export functions
export cleanup_old_trainings, export_training, import_training, analyze_training_evolution,
       backup_weights, validate_weights_file, weights_utilities_menu