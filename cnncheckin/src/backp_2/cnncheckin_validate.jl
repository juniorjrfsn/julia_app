module cnncheckin_validate

using JLD2
import .cnncheckin_core

# Função para carregar modelo, configuração e dados do modelo
function load_model_and_config(model_filepath::String, config_filepath::String)
    println("📂 Carregando modelo, configuração e dados do modelo...")
    
    config = cnncheckin_core.load_config(config_filepath)
    cnncheckin_core.validate_config(config)
    
    if !isfile(model_filepath)
        error("Arquivo do modelo não encontrado: $model_filepath")
    end
    
    try
        data = load(model_filepath)
        model_data = data["model_data"]
        model_state = model_data["model_state"]
        person_names = config["data"]["person_names"]
        num_classes = config["model"]["num_classes"]
        model_data_toml = cnncheckin_core.load_model_data_toml(cnncheckin_core.MODEL_DATA_TOML_PATH)
        
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
    
    if !isfile(cnncheckin_core.CONFIG_PATH)
        push!(errors, "Arquivo de configuração não encontrado: $cnncheckin_core.CONFIG_PATH")
    end
    if !isfile(cnncheckin_core.MODEL_PATH)
        push!(errors, "Arquivo do modelo não encontrado: $cnncheckin_core.MODEL_PATH")
    end
    if !isfile(cnncheckin_core.MODEL_DATA_TOML_PATH)
        push!(warnings, "Arquivo de dados do modelo não encontrado: $cnncheckin_core.MODEL_DATA_TOML_PATH")
    end
    
    if !isempty(errors)
        println("❌ Erros encontrados:")
        for error in errors
            println("   - $error")
        end
        return false
    end
    
    try
        config = cnncheckin_core.load_config(cnncheckin_core.CONFIG_PATH)
        cnncheckin_core.validate_config(config)
        println("✅ Configuração válida")
        
        model_data_toml = cnncheckin_core.load_model_data_toml(cnncheckin_core.MODEL_DATA_TOML_PATH)
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
                    person_name = cnncheckin_core.extract_person_name(filename)
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
            model, person_names, _, model_data_toml = load_model_and_config(cnncheckin_core.MODEL_PATH, cnncheckin_core.CONFIG_PATH)
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

# Executar comando
if abspath(PROGRAM_FILE) == @__FILE__
    success = validate_command()
    if success
        println("✅ Validação concluída!")
    else
        println("💥 Falha na validação")
    end
end

end # module cnncheckin_validate
