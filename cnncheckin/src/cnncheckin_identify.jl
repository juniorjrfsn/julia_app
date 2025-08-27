# projeto: cnncheckin
# file: cnncheckin/src/cnncheckin_identify.jl

using Flux
using JLD2
using Statistics

include("cnncheckin_core.jl")
using .CNNCheckinCore

# Function to load model, configuration and model data
function load_model_and_config(model_filepath::String, config_filepath::String)
    println("Carregando modelo, configuração e dados...")
    
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
        
        println("Modelo e configuração carregados com sucesso!")
        println("Informações do modelo:")
        println("   - Classes: $num_classes")
        println("   - Pessoas: $(join(person_names, ", "))")
        println("   - Acurácia: $(round(config["training"]["final_accuracy"]*100, digits=2))%")
        println("   - Criado em: $(config["data"]["timestamp"])")
        
        # Verify mapping
        println("Mapeamento correto:")
        for (i, name) in enumerate(person_names)
            println("   - Índice $i: $name")
        end
        
        if model_data_toml !== nothing
            println("   - Dados TOML disponíveis: Sim")
            total_params = get(get(model_data_toml, "weights_summary", Dict()), "total_parameters", 0)
            if total_params > 0
                println("   - Total de parâmetros: $(total_params)")
            end
        else
            println("   - Dados TOML disponíveis: Não")
        end
        
        return model_state, person_names, config, model_data_toml
    catch e
        error("Erro ao carregar modelo: $e")
    end
end

# Function to make prediction and log example
function predict_person(model, person_names, img_path::String; save_example::Bool = true)
    println("Processando imagem: $img_path")
    
    # Preprocess image
    img_arrays = CNNCheckinCore.preprocess_image(img_path; augment=false)
    
    if img_arrays === nothing || length(img_arrays) == 0
        println("Não foi possível processar a imagem")
        return nothing, 0.0
    end
    
    img_array = img_arrays[1]
    println("Dimensões da imagem processada: $(size(img_array))")
    
    # Prepare input tensor - ensuring correct format
    img_tensor = reshape(img_array, size(img_array)..., 1)
    println("Dimensões do tensor de entrada: $(size(img_tensor))")
    
    try
        println("Executando predição...")
        
        # Run model
        logits = model(img_tensor)
        println("Dimensões da saída do modelo: $(size(logits))")
        println("Logits brutos: $(vec(logits))")
        
        # Check dimension compatibility
        if size(logits, 1) != length(person_names)
            error("Dimensão de saída do modelo ($(size(logits, 1))) não corresponde ao número de classes ($(length(person_names)))")
        end
        
        # Convert to Float32 and apply softmax robustly
        logits_vec = Float32.(vec(logits))
        println("Logits como vetor Float32: $logits_vec")
        
        # Apply softmax manually for better control
        max_logit = maximum(logits_vec)
        exp_logits = exp.(logits_vec .- max_logit)
        sum_exp = sum(exp_logits)
        probabilities = exp_logits ./ sum_exp
        
        println("Probabilidades: $probabilities")
        
        # Show probability for each person
        println("Probabilidades por pessoa:")
        for (i, (name, prob)) in enumerate(zip(person_names, probabilities))
            println("   $i. $name: $(round(prob*100, digits=2))%")
        end
        
        # Find class with highest probability
        pred_class = argmax(probabilities)
        confidence = probabilities[pred_class]
        
        println("Classe predita: $pred_class")
        println("Confiança: $(round(confidence*100, digits=2))%")
        
        # Check if index is valid
        if pred_class <= 0 || pred_class > length(person_names)
            println("Índice de classe inválido: $pred_class")
            return "Desconhecido", Float64(confidence)
        end
        
        # The argmax already returns the correct index for Julia (1-based)
        person_name = person_names[pred_class]
        println("Pessoa identificada: $person_name")
        
        # Save example if requested
        if save_example
            try
                CNNCheckinCore.add_prediction_example_to_toml(img_path, person_name, Float64(confidence))
                println("Exemplo salvo com sucesso")
            catch e
                println("Erro ao salvar exemplo: $e")
            end
        end
        
        return person_name, Float64(confidence)
        
    catch e
        println("Erro ao realizar predição: $e")
        println("Detalhes do erro:")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
        return nothing, 0.0
    end
end

# Function to display prediction results in a formatted way
function display_prediction_result(person_name, confidence, img_path::String)
    println("\n" * "="^60)
    println("RESULTADO DA IDENTIFICAÇÃO FACIAL")
    println("="^60)
    println("📸 Imagem: $(basename(img_path))")
    println("👤 Pessoa identificada: $person_name")
    println("📊 Confiança: $(round(confidence*100, digits=2))%")
    
    # Confidence level assessment
    if confidence >= 0.9
        println("✅ Confiança: MUITO ALTA")
    elseif confidence >= 0.7
        println("⚡ Confiança: ALTA")
    elseif confidence >= 0.5
        println("⚠️  Confiança: MODERADA")
    else
        println("❌ Confiança: BAIXA - Verificar manualmente")
    end
    
    println("🕐 Timestamp: $(Dates.format(Dates.now(), "dd/mm/yyyy HH:MM:SS"))")
    println("="^60)
end

# Function to validate authentication image against known person
function authenticate_person(model, person_names, img_path::String, expected_person::String; 
                           confidence_threshold::Float64 = 0.7)
    println("🔐 Autenticando pessoa: $expected_person")
    
    predicted_person, confidence = predict_person(model, person_names, img_path; save_example=false)
    
    if predicted_person === nothing
        return false, 0.0, "Erro na predição"
    end
    
    is_authenticated = (predicted_person == expected_person) && (confidence >= confidence_threshold)
    
    if is_authenticated
        status = "✅ AUTENTICADO"
    elseif predicted_person != expected_person
        status = "❌ PESSOA INCORRETA (predito: $predicted_person)"
    else
        status = "❌ CONFIANÇA INSUFICIENTE ($(round(confidence*100, digits=2))% < $(round(confidence_threshold*100, digits=0))%)"
    end
    
    println("🔍 Resultado da autenticação:")
    println("   - Esperado: $expected_person")
    println("   - Predito: $predicted_person")
    println("   - Confiança: $(round(confidence*100, digits=2))%")
    println("   - Status: $status")
    
    return is_authenticated, confidence, status
end

# Batch identification function for multiple images
function batch_identify(model, person_names, image_directory::String; 
                       output_file::String = "batch_identification_results.txt")
    println("🔄 Iniciando identificação em lote...")
    
    if !isdir(image_directory)
        error("Diretório não encontrado: $image_directory")
    end
    
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    results = []
    
    image_files = filter(f -> lowercase(splitext(f)[2]) in image_extensions, readdir(image_directory))
    
    if length(image_files) == 0
        println("❌ Nenhuma imagem encontrada no diretório: $image_directory")
        return results
    end
    
    println("📁 Encontradas $(length(image_files)) imagens para processar")
    
    for (i, filename) in enumerate(image_files)
        img_path = joinpath(image_directory, filename)
        println("\n[$i/$(length(image_files))] Processando: $filename")
        
        try
            person_name, confidence = predict_person(model, person_names, img_path; save_example=true)
            
            result = Dict(
                "filename" => filename,
                "path" => img_path,
                "predicted_person" => person_name,
                "confidence" => confidence,
                "timestamp" => string(Dates.now()),
                "success" => person_name !== nothing
            )
            
            push!(results, result)
            
            if person_name !== nothing
                println("   ✅ $(person_name) - $(round(confidence*100, digits=2))%")
            else
                println("   ❌ Falha na identificação")
            end
            
        catch e
            println("   ❌ Erro ao processar $filename: $e")
            result = Dict(
                "filename" => filename,
                "path" => img_path,
                "predicted_person" => nothing,
                "confidence" => 0.0,
                "timestamp" => string(Dates.now()),
                "success" => false,
                "error" => string(e)
            )
            push!(results, result)
        end
    end
    
    # Save results to file
    try
        open(output_file, "w") do io
            println(io, "RESULTADO DA IDENTIFICAÇÃO EM LOTE")
            println(io, "Gerado em: $(Dates.format(Dates.now(), "dd/mm/yyyy HH:MM:SS"))")
            println(io, "Diretório: $image_directory")
            println(io, "Total de imagens: $(length(image_files))")
            println(io, "="^80)
            
            for result in results
                println(io, "\nArquivo: $(result["filename"])")
                println(io, "Pessoa: $(result["predicted_person"])")
                println(io, "Confiança: $(round(result["confidence"]*100, digits=2))%")
                println(io, "Status: $(result["success"] ? "Sucesso" : "Falha")")
                if haskey(result, "error")
                    println(io, "Erro: $(result["error"])")
                end
                println(io, "-"^40)
            end
        end
        println("\n💾 Resultados salvos em: $output_file")
    catch e
        println("❌ Erro ao salvar resultados: $e")
    end
    
    # Summary
    successful = sum(r["success"] for r in results)
    println("\n📊 RESUMO DA IDENTIFICAÇÃO EM LOTE:")
    println("   - Total de imagens: $(length(image_files))")
    println("   - Sucessos: $successful")
    println("   - Falhas: $(length(image_files) - successful)")
    println("   - Taxa de sucesso: $(round(successful/length(image_files)*100, digits=1))%")
    
    return results
end

# Main identification command
function identify_command(img_path::String; auth_mode::Bool = false, expected_person::String = "")
    println("🤖 Sistema de Reconhecimento Facial - Modo Identificação")
    
    if !isfile(img_path)
        error("Arquivo de imagem não encontrado: $img_path")
    end
    
    try
        # Load model and configuration
        model, person_names, config, model_data_toml = load_model_and_config(
            CNNCheckinCore.MODEL_PATH, 
            CNNCheckinCore.CONFIG_PATH
        )
        
        if auth_mode && !isempty(expected_person)
            # Authentication mode
            is_authenticated, confidence, status = authenticate_person(model, person_names, 
                                                                      img_path, expected_person)
            return is_authenticated, confidence, status
        else
            # Regular identification mode
            person_name, confidence = predict_person(model, person_names, img_path)
            
            if person_name !== nothing
                display_prediction_result(person_name, confidence, img_path)
                return person_name, confidence
            else
                println("❌ Falha na identificação da imagem")
                return nothing, 0.0
            end
        end
        
    catch e
        println("❌ Erro durante identificação: $e")
        return nothing, 0.0
    end
end

# Command line interface
function main()
    if length(ARGS) == 0
        println("Uso:")
        println("  julia cnncheckin_identify.jl <caminho_da_imagem>")
        println("  julia cnncheckin_identify.jl <caminho_da_imagem> --auth <nome_esperado>")
        println("  julia cnncheckin_identify.jl --batch <diretório_imagens>")
        println()
        println("Exemplos:")
        println("  julia cnncheckin_identify.jl foto.jpg")
        println("  julia cnncheckin_identify.jl foto.jpg --auth \"João Silva\"")
        println("  julia cnncheckin_identify.jl --batch ./fotos_teste/")
        return
    end
    
    if ARGS[1] == "--batch"
        if length(ARGS) < 2
            println("❌ Especifique o diretório para identificação em lote")
            return
        end
        
        # Load model first
        try
            model, person_names, config, model_data_toml = load_model_and_config(
                CNNCheckinCore.MODEL_PATH, 
                CNNCheckinCore.CONFIG_PATH
            )
            
            batch_identify(model, person_names, ARGS[2])
        catch e
            println("❌ Erro durante identificação em lote: $e")
        end
        
    elseif length(ARGS) >= 3 && ARGS[2] == "--auth"
        # Authentication mode
        img_path = ARGS[1]
        expected_person = ARGS[3]
        
        is_authenticated, confidence, status = identify_command(img_path; 
                                                               auth_mode=true, 
                                                               expected_person=expected_person)
        
        if is_authenticated
            println("\n🎉 Autenticação bem-sucedida!")
        else
            println("\n🚫 Autenticação falhou!")
        end
        
    else
        # Regular identification mode
        img_path = ARGS[1]
        result = identify_command(img_path)
        
        if result[1] !== nothing
            println("\n🎯 Identificação concluída com sucesso!")
        else
            println("\n❌ Identificação não foi possível")
        end
    end
end

# Execute if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end




# Identificação simples: julia cnncheckin_identify.jl foto.jpg
# Autenticação: julia cnncheckin_identify.jl foto.jpg --auth "Nome Pessoa"
# Lote: julia cnncheckin_identify.jl --batch ./diretorio/

# Identificação simples: julia cnncheckin_identify.jl ../../../dados/fotos_auth/nl.jpg

#  julia cnncheckin_identify.jl ../../../dados/fotos_auth/objeto-1.jpg
#  julia cnncheckin_identify.jl ../../../dados/fotos_auth/objeto-2.jpg
#  julia cnncheckin_identify.jl ../../../dados/fotos_auth/objeto-3.jpeg

#  julia cnncheckin_identify.jl ../../../dados/fotos_auth/teste.png