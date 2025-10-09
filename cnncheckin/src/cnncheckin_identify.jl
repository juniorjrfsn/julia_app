# projeto: cnncheckin
# file: cnncheckin/src/cnncheckin_identify.jl

 # projeto: cnncheckin
# file: cnncheckin/src/cnncheckin_identify.jl
# descrição: Script para identificação de pessoas usando modelo treinado

using Flux
using JLD2
using Statistics
using Dates
using Logging

include("cnncheckin_core.jl")
using .CNNCheckinCore

# ============================================================================
# CARREGAMENTO DO MODELO
# ============================================================================

"""
    load_model_for_inference()
        -> Tuple{Chain, Vector{String}, Dict, Union{Dict, Nothing}}

Carrega o modelo treinado e suas configurações para inferência.

# Retorna
- Modelo CNN
- Lista de nomes das pessoas
- Configuração do sistema
- Metadados do modelo (ou nothing)
"""
function load_model_for_inference()
    @info "📂 Carregando modelo para inferência..."
    
    # Carregar configuração
    config = CNNCheckinCore.load_config(CNNCheckinCore.CONFIG_PATH)
    CNNCheckinCore.validate_config(config)
    
    # Verificar existência do modelo
    if !isfile(CNNCheckinCore.MODEL_PATH)
        throw(ArgumentError("Modelo não encontrado: $(CNNCheckinCore.MODEL_PATH)"))
    end
    
    # Carregar modelo
    try
        data = load(CNNCheckinCore.MODEL_PATH)
        model_data = data["model_data"]
        model = model_data["model_state"]
        person_names = config["data"]["person_names"]
        num_classes = config["model"]["num_classes"]
        
        # Carregar metadados
        model_metadata = CNNCheckinCore.load_model_data_toml(CNNCheckinCore.MODEL_DATA_TOML_PATH)
        
        @info """
        ✅ Modelo carregado com sucesso!
           - Classes: $num_classes
           - Pessoas: $(join(person_names, ", "))
           - Acurácia: $(round(config["training"]["final_accuracy"]*100, digits=2))%
           - Criado: $(config["data"]["timestamp"])
        """
        
        # Verificar mapeamento
        @info "🗂️  Mapeamento pessoa → label:"
        for (i, name) in enumerate(person_names)
            @info "   $i: $name"
        end
        
        return model, person_names, config, model_metadata
        
    catch e
        throw(ErrorException("Erro ao carregar modelo: $e"))
    end
end

# ============================================================================
# PREDIÇÃO
# ============================================================================

"""
    predict_person(model, person_names::Vector{String}, img_path::String; 
                  save_example::Bool=true)
        -> Tuple{Union{String, Nothing}, Float64}

Realiza predição de pessoa em uma imagem.

# Argumentos
- `model`: Modelo CNN treinado
- `person_names`: Lista de nomes das pessoas
- `img_path`: Caminho da imagem
- `save_example`: Se deve salvar exemplo nos metadados

# Retorna
- Nome da pessoa identificada (ou nothing se erro)
- Confiança da predição (0.0 a 1.0)
"""
function predict_person(model, person_names::Vector{String}, img_path::String; 
                       save_example::Bool=true)
    @info "🔍 Processando imagem..." path=img_path
    
    # Preprocessar imagem
    img_arrays = CNNCheckinCore.preprocess_image(img_path; augment=false)
    
    if img_arrays === nothing || isempty(img_arrays)
        @error "Não foi possível processar a imagem"
        return nothing, 0.0
    end
    
    img_array = img_arrays[1]
    @debug "Dimensões da imagem processada: $(size(img_array))"
    
    # Preparar tensor de entrada
    img_tensor = reshape(img_array, size(img_array)..., 1)
    
    try
        # Executar modelo
        logits = model(img_tensor)
        @debug "Logits: $(vec(logits))"
        
        # Verificar compatibilidade de dimensões
        if size(logits, 1) != length(person_names)
            throw(DimensionMismatch(
                "Saída do modelo ($(size(logits, 1))) não corresponde ao número de classes ($(length(person_names)))"
            ))
        end
        
        # Aplicar softmax manualmente para melhor controle
        logits_vec = Float32.(vec(logits))
        max_logit = maximum(logits_vec)
        exp_logits = exp.(logits_vec .- max_logit)
        probabilities = exp_logits ./ sum(exp_logits)
        
        @debug "Probabilidades: $probabilities"
        
        # Mostrar probabilidades por pessoa
        @info "📊 Probabilidades por pessoa:"
        for (i, (name, prob)) in enumerate(zip(person_names, probabilities))
            @info "   $i. $name: $(round(prob*100, digits=2))%"
        end
        
        # Encontrar classe com maior probabilidade
        pred_class = argmax(probabilities)
        confidence = probabilities[pred_class]
        
        # Validar índice
        if pred_class <= 0 || pred_class > length(person_names)
            @error "Índice de classe inválido: $pred_class"
            return "Desconhecido", Float64(confidence)
        end
        
        person_name = person_names[pred_class]
        @info "✅ Pessoa identificada: $person_name ($(round(confidence*100, digits=2))%)"
        
        # Salvar exemplo se solicitado
        if save_example
            try
                CNNCheckinCore.add_prediction_example_to_toml(
                    img_path, 
                    person_name, 
                    Float64(confidence)
                )
            catch e
                @debug "Não foi possível salvar exemplo" exception=e
            end
        end
        
        return person_name, Float64(confidence)
        
    catch e
        @error "Erro durante predição" exception=(e, catch_backtrace())
        return nothing, 0.0
    end
end

"""
    display_prediction_result(person_name::String, confidence::Float64, img_path::String)

Exibe o resultado da predição de forma formatada.
"""
function display_prediction_result(person_name::String, confidence::Float64, img_path::String)
    println("\n" * "="^70)
    println("🎯 RESULTADO DA IDENTIFICAÇÃO FACIAL")
    println("="^70)
    println("📸 Imagem: $(basename(img_path))")
    println("👤 Pessoa identificada: $person_name")
    println("📊 Confiança: $(round(confidence*100, digits=2))%")
    
    # Avaliação do nível de confiança
    confidence_level = if confidence >= 0.9
        "✅ MUITO ALTA"
    elseif confidence >= 0.7
        "⚡ ALTA"
    elseif confidence >= 0.5
        "⚠️  MODERADA"
    else
        "❌ BAIXA - Verificar manualmente"
    end
    
    println("🔒 Nível de confiança: $confidence_level")
    println("🕐 Timestamp: $(Dates.format(Dates.now(), "dd/mm/yyyy HH:MM:SS"))")
    println("="^70 * "\n")
end

# ============================================================================
# AUTENTICAÇÃO
# ============================================================================

"""
    authenticate_person(model, person_names::Vector{String}, img_path::String, 
                       expected_person::String; confidence_threshold::Float64=0.7)
        -> Tuple{Bool, Float64, String}

Autentica se uma imagem corresponde a uma pessoa esperada.

# Retorna
- Se autenticação foi bem-sucedida
- Confiança da predição
- Status da autenticação
"""
function authenticate_person(model, person_names::Vector{String}, img_path::String, 
                            expected_person::String; confidence_threshold::Float64=0.7)
    @info "🔐 Autenticando pessoa..." expected=expected_person threshold=confidence_threshold
    
    predicted_person, confidence = predict_person(model, person_names, img_path; save_example=false)
    
    if predicted_person === nothing
        return false, 0.0, "❌ Erro na predição"
    end
    
    is_authenticated = (predicted_person == expected_person) && (confidence >= confidence_threshold)
    
    status = if is_authenticated
        "✅ AUTENTICADO"
    elseif predicted_person != expected_person
        "❌ PESSOA INCORRETA (predito: $predicted_person)"
    else
        "❌ CONFIANÇA INSUFICIENTE ($(round(confidence*100, digits=2))% < $(round(confidence_threshold*100, digits=0))%)"
    end
    
    @info """
    🔍 Resultado da autenticação:
       - Esperado: $expected_person
       - Predito: $predicted_person
       - Confiança: $(round(confidence*100, digits=2))%
       - Status: $status
    """
    
    return is_authenticated, confidence, status
end

# ============================================================================
# IDENTIFICAÇÃO EM LOTE
# ============================================================================

"""
    batch_identify(model, person_names::Vector{String}, image_directory::String;
                  output_file::String="batch_results.txt")
        -> Vector{Dict}

Processa múltiplas imagens de um diretório.

# Retorna
Vector de dicionários com resultados de cada imagem
"""
function batch_identify(model, person_names::Vector{String}, image_directory::String;
                       output_file::String="batch_results.txt")
    @info "📁 Iniciando identificação em lote..." directory=image_directory
    
    if !isdir(image_directory)
        throw(ArgumentError("Diretório não encontrado: $image_directory"))
    end
    
    # Encontrar imagens
    image_files = filter(
        f -> lowercase(splitext(f)[2]) in CNNCheckinCore.VALID_IMAGE_EXTENSIONS,
        readdir(image_directory)
    )
    
    if isempty(image_files)
        @warn "Nenhuma imagem encontrada no diretório"
        return Dict[]
    end
    
    @info "📊 Encontradas $(length(image_files)) imagens para processar"
    
    results = Dict[]
    
    for (i, filename) in enumerate(image_files)
        img_path = joinpath(image_directory, filename)
        @info "[$i/$(length(image_files))] Processando: $filename"
        
        try
            person_name, confidence = predict_person(
                model, 
                person_names, 
                img_path; 
                save_example=true
            )
            
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
                @info "   ✅ $(person_name) - $(round(confidence*100, digits=2))%"
            else
                @warn "   ❌ Falha na identificação"
            end
            
        catch e
            @error "Erro ao processar" filename=filename exception=(e, catch_backtrace())
            
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
    
    # Salvar resultados
    try
        open(output_file, "w") do io
            println(io, "="^80)
            println(io, "RESULTADO DA IDENTIFICAÇÃO EM LOTE")
            println(io, "="^80)
            println(io, "Gerado em: $(Dates.format(Dates.now(), "dd/mm/yyyy HH:MM:SS"))")
            println(io, "Diretório: $image_directory")
            println(io, "Total de imagens: $(length(image_files))")
            println(io, "="^80 * "\n")
            
            for result in results
                println(io, "Arquivo: $(result["filename"])")
                println(io, "Pessoa: $(result["predicted_person"])")
                println(io, "Confiança: $(round(result["confidence"]*100, digits=2))%")
                println(io, "Status: $(result["success"] ? "Sucesso" : "Falha")")
                if haskey(result, "error")
                    println(io, "Erro: $(result["error"])")
                end
                println(io, "-"^40)
            end
        end
        
        @info "💾 Resultados salvos: $output_file"
        
    catch e
        @error "Erro ao salvar resultados" exception=(e, catch_backtrace())
    end
    
    # Exibir resumo
    successful = count(r -> r["success"], results)
    
    println("\n" * "="^70)
    println("📊 RESUMO DA IDENTIFICAÇÃO EM LOTE")
    println("="^70)
    println("Total de imagens: $(length(image_files))")
    println("Sucessos: $successful")
    println("Falhas: $(length(image_files) - successful)")
    println("Taxa de sucesso: $(round(successful/length(image_files)*100, digits=1))%")
    println("="^70 * "\n")
    
    return results
end

# ============================================================================
# FUNÇÕES DE COMANDO
# ============================================================================

"""
    identify_command(img_path::String; auth_mode::Bool=false, expected_person::String="")
        -> Tuple{Union{String, Bool, Nothing}, Float64, String}

Executa comando de identificação ou autenticação.
"""
function identify_command(img_path::String; auth_mode::Bool=false, expected_person::String="")
    println("\n" * "="^70)
    println("🤖 SISTEMA DE RECONHECIMENTO FACIAL - IDENTIFICAÇÃO")
    println("="^70 * "\n")
    
    if !isfile(img_path)
        throw(ArgumentError("Arquivo de imagem não encontrado: $img_path"))
    end
    
    try
        # Carregar modelo
        model, person_names, config, model_metadata = load_model_for_inference()
        
        if auth_mode && !isempty(expected_person)
            # Modo autenticação
            is_authenticated, confidence, status = authenticate_person(
                model, 
                person_names, 
                img_path, 
                expected_person
            )
            
            return is_authenticated, confidence, status
            
        else
            # Modo identificação
            person_name, confidence = predict_person(model, person_names, img_path)
            
            if person_name !== nothing
                display_prediction_result(person_name, confidence, img_path)
                return person_name, confidence, "success"
            else
                @error "Falha na identificação da imagem"
                return nothing, 0.0, "error"
            end
        end
        
    catch e
        @error "Erro durante identificação" exception=(e, catch_backtrace())
        return nothing, 0.0, "error"
    end
end

"""
    batch_command(image_directory::String)

Executa comando de identificação em lote.
"""
function batch_command(image_directory::String)
    try
        model, person_names, config, model_metadata = load_model_for_inference()
        batch_identify(model, person_names, image_directory)
        
    catch e
        @error "Erro durante identificação em lote" exception=(e, catch_backtrace())
    end
end

# ============================================================================
# INTERFACE DE LINHA DE COMANDO
# ============================================================================

"""
    main()

Função principal para execução via linha de comando.
"""
function main()
    if length(ARGS) == 0
        println("""
        Uso:
          julia cnncheckin_identify.jl <caminho_da_imagem>
          julia cnncheckin_identify.jl <caminho_da_imagem> --auth <nome_esperado>
          julia cnncheckin_identify.jl --batch <diretório_imagens>
        
        Exemplos:
          julia cnncheckin_identify.jl foto.jpg
          julia cnncheckin_identify.jl foto.jpg --auth "João Silva"
          julia cnncheckin_identify.jl --batch ./fotos_teste/
        """)
        return
    end
    
    if ARGS[1] == "--batch"
        if length(ARGS) < 2
            @error "Especifique o diretório para identificação em lote"
            return
        end
        
        batch_command(ARGS[2])
        
    elseif length(ARGS) >= 3 && ARGS[2] == "--auth"
        # Modo autenticação
        img_path = ARGS[1]
        expected_person = ARGS[3]
        
        is_authenticated, confidence, status = identify_command(
            img_path; 
            auth_mode=true, 
            expected_person=expected_person
        )
        
        if is_authenticated
            println("\n🎉 Autenticação bem-sucedida!")
        else
            println("\n🚫 Autenticação falhou!")
        end
        
    else
        # Modo identificação
        img_path = ARGS[1]
        result, confidence, status = identify_command(img_path)
        
        if result !== nothing
            println("✅ Identificação concluída com sucesso!")
        else
            println("❌ Identificação não foi possível")
        end
    end
end

# ============================================================================
# EXECUÇÃO
# ============================================================================

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