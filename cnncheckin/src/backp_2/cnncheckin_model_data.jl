# cnncheckin_model_data.jl
using Statistics
include("cnncheckin_core.jl")
using .CNNCheckinCore

# Função para exibir detalhes dos dados do modelo
function model_data_command()
    println("🔬 Dados Detalhados do Modelo")
    
    model_data_toml = CNNCheckinCore.load_model_data_toml(CNNCheckinCore.MODEL_DATA_TOML_PATH)
    if model_data_toml === nothing
        println("❌ Dados do modelo não encontrados. Execute primeiro: julia cnncheckin_train.jl")
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
            println("\n🔍 Histórico de Predições ($(length(examples)) total):")
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
            println("\n   🕒 Últimas predições:")
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

# Executar comando se for chamado diretamente
if abspath(PROGRAM_FILE) == @__FILE__
    success = model_data_command()
    if success
        println("✅ Dados técnicos exibidos!")
    else
        println("💥 Falha ao obter dados do modelo")
    end
end

# ---

# cnncheckin_validate.jl
using JLD2
include("cnncheckin_core.jl")
using .CNNCheckinCore

# Função para carregar modelo, configuração e dados do modelo
function load_model_and_config(model_filepath::String, config_filepath::String)
    println("📂 Carregando modelo, configuração e dados do modelo...")
    
    config = CNNCheckinCore.load_config(config_filepath)
    CNNCheckinCore.validate_config(config)
    
    if !isfile(model_filepath)
        error("Arquivo do modelo não encontrado: $model_filepath")
    end
    
    try
        data = load(model_filepath)
        model_data = data["model_data"]
        model_state = model_data["model_state"]
        person_names = config["data"]["person_names"]
        num_classes = config["model"]["num_classes"]
        model_data_toml = CNNCheckinCore.load_model_data_toml(CNNCheckinCore.MODEL_DATA_TOML_PATH)
        
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

# Função para validar modelo, configuração e dados
function validate_command()
    println("🔍 Validando modelo, configuração e dados...")
    
    errors = []
    warnings = []
    
    if !isfile(CNNCheckinCore.CONFIG_PATH)
        push!(errors, "Arquivo de configuração não encontrado: $(CNNCheckinCore.CONFIG_PATH)")
    end
    if !isfile(CNNCheckinCore.MODEL_PATH)
        push!(errors, "Arquivo do modelo não encontrado: $(CNNCheckinCore.MODEL_PATH)")
    end
    if !isfile(CNNCheckinCore.MODEL_DATA_TOML_PATH)
        push!(warnings, "Arquivo de dados do modelo não encontrado: $(CNNCheckinCore.MODEL_DATA_TOML_PATH)")
    end
    
    if !isempty(errors)
        println("❌ Erros encontrados:")
        for error in errors
            println("   - $error")
        end
        return false
    end
    
    try
        config = CNNCheckinCore.load_config(CNNCheckinCore.CONFIG_PATH)
        CNNCheckinCore.validate_config(config)
        println("✅ Configuração válida")
        
        model_data_toml = CNNCheckinCore.load_model_data_toml(CNNCheckinCore.MODEL_DATA_TOML_PATH)
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
                    person_name = CNNCheckinCore.extract_person_name(filename)
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
            model, person_names, _, model_data_toml = load_model_and_config(CNNCheckinCore.MODEL_PATH, CNNCheckinCore.CONFIG_PATH)
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

# Executar comando se for chamado diretamente
if abspath(PROGRAM_FILE) == @__FILE__
    success = validate_command()
    if success
        println("✅ Validação concluída!")
    else
        println("💥 Falha na validação")
    end
end

# ---

# cnncheckin_export_config.jl
using Dates
include("cnncheckin_core.jl")
using .CNNCheckinCore

# Função para exportar configuração
function export_config_command(output_path::String = "modelo_config_export.toml")
    println("📤 Exportando configuração do modelo...")
    
    if !isfile(CNNCheckinCore.CONFIG_PATH)
        println("❌ Configuração não encontrada. Execute primeiro: julia cnncheckin_train.jl")
        return false
    end
    
    try
        config = CNNCheckinCore.load_config(CNNCheckinCore.CONFIG_PATH)
        config["export"] = Dict(
            "exported_at" => string(Dates.now()),
            "exported_from" => CNNCheckinCore.CONFIG_PATH,
            "export_version" => "1.0"
        )
        success = CNNCheckinCore.save_config(config, output_path)
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

# Executar comando se for chamado diretamente
if abspath(PROGRAM_FILE) == @__FILE__
    output_path = length(ARGS) >= 1 ? ARGS[1] : "modelo_config_export.toml"
    success = export_config_command(output_path)
    if success
        println("✅ Exportação de configuração concluída!")
    else
        println("💥 Falha na exportação")
    end
end