# projeto: cnncheckin
# file: cnncheckin/src/checkin/identif.jl
# descrição: Script para identificação de pessoas usando modelo treinado

# projeto: cnncheckin
# file: cnncheckin/src/checkin/identif.jl
# descrição: Script para identificação de pessoas usando modelo treinado

module Identif
    using Flux
    using JLD2
    using Statistics
    using Dates
    using Logging
    using Images
    using VideoIO
    using FileIO

    include("config_lib.jl") # inclui o módulo ConfigLib
    using ..CNNCheckinCore  # Acessa o módulo pai (Checkin) que já incluiu CNNCheckinCore
    using ..Menu  # Importa o módulo Menu

    # ============================================================================
    # CARREGAMENTO DO MODELO
    # ============================================================================

    """
        load_model_for_inference()
            -> Tuple{Chain, Vector{String}, Dict, Union{Dict, Nothing}}

    Carrega o modelo treinado e suas configurações para inferência.
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
            @info "🗂️ Mapeamento pessoa → label:"
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
        predict_from_array(model, person_names::Vector{String}, img_array::Array{Float32, 3})
            -> Tuple{Union{String, Nothing}, Float64}

    Realiza predição de pessoa a partir de um array de imagem já processado.
    Versão otimizada para uso em tempo real (webcam).
    """
    function predict_from_array(model, person_names::Vector{String}, img_array::Array{Float32, 3})
        try
            # Preparar tensor de entrada
            img_tensor = reshape(img_array, size(img_array)..., 1)
            
            # Executar modelo
            logits = model(img_tensor)
            
            # Verificar compatibilidade de dimensões
            if size(logits, 1) != length(person_names)
                return "Desconhecido", 0.0
            end
            
            # Aplicar softmax
            logits_vec = Float32.(vec(logits))
            max_logit = maximum(logits_vec)
            exp_logits = exp.(logits_vec .- max_logit)
            probabilities = exp_logits ./ sum(exp_logits)
            
            # Encontrar classe com maior probabilidade
            pred_class = argmax(probabilities)
            confidence = probabilities[pred_class]
            
            # Validar índice
            if pred_class <= 0 || pred_class > length(person_names)
                return "Desconhecido", Float64(confidence)
            end
            
            person_name = person_names[pred_class]
            return person_name, Float64(confidence)
            
        catch e
            @debug "Erro durante predição de array" exception=e
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
            "⚠️ MODERADA"
        else
            "❌ BAIXA - Verificar manualmente"
        end
        
        println("🔒 Nível de confiança: $confidence_level")
        println("🕐 Timestamp: $(Dates.format(Dates.now(), "dd/mm/yyyy HH:MM:SS"))")
        println("="^70 * "\n")
    end

    # ============================================================================
    # IDENTIFICAÇÃO POR WEBCAM
    # ============================================================================

    """
        process_frame_for_prediction(frame) -> Union{Array{Float32, 3}, Nothing}

    Processa um frame da webcam para predição.
    """
    function process_frame_for_prediction(frame)
        try
            # Converter frame para RGB se necessário
            img = CNNCheckinCore.convert_to_rgb(frame)
            
            # Redimensionar para o tamanho esperado pelo modelo
            img_resized = imresize(img, CNNCheckinCore.IMG_SIZE)
            
            # Converter para array Float32
            img_array = Float32.(channelview(img_resized))
            img_array = permutedims(img_array, (2, 3, 1))
            
            # Normalizar
            img_array = CNNCheckinCore.normalize_image(img_array)
            
            return img_array
            
        catch e
            @debug "Erro ao processar frame" exception=e
            return nothing
        end
    end

    """
        save_webcam_capture(frame, person_name::String, confidence::Float64) -> String

    Salva um frame capturado da webcam.
    """
    function save_webcam_capture(frame, person_name::String, confidence::Float64)
        try
            # Criar diretório de capturas se não existir
            captures_dir = joinpath(CNNCheckinCore.AUTH_DATA_PATH, "webcam_captures")
            if !isdir(captures_dir)
                mkpath(captures_dir)
            end
            
            # Nome do arquivo com timestamp
            timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
            filename = "$(person_name)_$(timestamp)_$(round(confidence*100, digits=0))pct.jpg"
            filepath = joinpath(captures_dir, filename)
            
            # Salvar imagem
            save(filepath, frame)
            
            return filepath
            
        catch e
            @error "Erro ao salvar captura" exception=e
            return ""
        end
    end

    """
        identify_from_webcam(model, person_names::Vector{String})

    Identifica pessoa usando webcam em tempo real.
    """
    function identify_from_webcam(model, person_names::Vector{String})
        println("\n" * "="^70)
        println("🎥 IDENTIFICAÇÃO POR WEBCAM")
        println("="^70)
        
        # Verificar se VideoIO está disponível
        try
            # Tentar abrir a webcam
            println("📷 Tentando abrir webcam...")
            
            # Configurações
            confidence_threshold = 0.7
            frame_skip = 5  # Processar a cada N frames para melhor performance
            frame_counter = 0
            last_prediction = ("", 0.0)
            save_captures = false
            
            println("\n⚙️ Configurações:")
            println("   • Limite de confiança: $(confidence_threshold*100)%")
            println("   • Processamento: 1 a cada $frame_skip frames")
            
            print("\n💾 Deseja salvar capturas automaticamente? (s/N): ")
            flush(stdout)
            save_input = strip(lowercase(readline()))
            save_captures = (save_input == "s")
            
            println("\n📋 Controles:")
            println("   • ESPAÇO - Capturar frame e salvar")
            println("   • C - Alternar salvamento automático")
            println("   • Q - Sair")
            println("   • + - Aumentar confiança mínima")
            println("   • - - Diminuir confiança mínima")
            
            println("\n🔄 Iniciando captura... (Pressione Q para sair)")
            println("="^70)
            
            # Abrir webcam (device 0 é geralmente a webcam padrão)
            try
                cam = VideoIO.opencamera()
                
                println("✅ Webcam conectada!")
                println("🎬 Iniciando identificação em tempo real...\n")
                
                while true
                    try
                        # Ler frame
                        frame = read(cam)
                        frame_counter += 1
                        
                        # Processar apenas a cada N frames
                        if frame_counter % frame_skip == 0
                            # Processar frame
                            img_array = process_frame_for_prediction(frame)
                            
                            if img_array !== nothing
                                # Fazer predição
                                person_name, confidence = predict_from_array(model, person_names, img_array)
                                
                                if person_name !== nothing && confidence >= confidence_threshold
                                    # Atualizar última predição
                                    last_prediction = (person_name, confidence)
                                    
                                    # Mostrar resultado
                                    conf_pct = round(confidence*100, digits=1)
                                    status = confidence >= 0.9 ? "✅" : confidence >= 0.8 ? "⚡" : "⚠️"
                                    println("$status $(Dates.format(Dates.now(), "HH:MM:SS")) | $person_name ($conf_pct%)")
                                    
                                    # Salvar captura se habilitado
                                    if save_captures
                                        filepath = save_webcam_capture(frame, person_name, confidence)
                                        if !isempty(filepath)
                                            println("   💾 Salvo: $(basename(filepath))")
                                        end
                                    end
                                end
                            end
                        end
                        
                        # Simular checagem de entrada de teclado (simplificado)
                        # Em uma implementação real, você usaria uma biblioteca para input não-bloqueante
                        
                    catch frame_error
                        if isa(frame_error, EOFError)
                            println("\n⚠️ Fim do stream da webcam")
                            break
                        else
                            @debug "Erro ao processar frame" exception=frame_error
                        end
                    end
                end
                
                close(cam)
                
            catch cam_error
                println("\n❌ Erro ao abrir webcam!")
                println("Possíveis causas:")
                println("   • Webcam não conectada ou em uso por outro programa")
                println("   • Permissões de acesso à câmera negadas")
                println("   • Driver da webcam não instalado")
                println("\nDetalhes: $cam_error")
                
                println("\n💡 Alternativa: Use a opção de arquivo de imagem")
                return
            end
            
            println("\n" * "="^70)
            println("🏁 Captura encerrada")
            
            if last_prediction[1] != ""
                println("\n📊 Última identificação:")
                println("   👤 Pessoa: $(last_prediction[1])")
                println("   📈 Confiança: $(round(last_prediction[2]*100, digits=2))%")
            end
            
            println("="^70 * "\n")
            
        catch e
            println("\n❌ ERRO: Funcionalidade de webcam não está totalmente disponível")
            println("\nDetalhes técnicos:")
            println("   $(typeof(e)): $e")
            
            println("\n🔧 Soluções:")
            println("   1. Certifique-se de que o VideoIO.jl está instalado:")
            println("      using Pkg; Pkg.add(\"VideoIO\")")
            println("   2. Verifique se a webcam está funcionando em outros programas")
            println("   3. No Linux, você pode precisar de:")
            println("      sudo apt-get install ffmpeg v4l-utils")
            println("   4. No Windows, certifique-se de que os drivers da webcam estão atualizados")
            
            println("\n💡 Por enquanto, use a opção de identificação por arquivo")
            
            print("\nPressione ENTER para continuar...")
            readline()
        end
    end

    # ============================================================================
    # AUTENTICAÇÃO
    # ============================================================================

    """
        authenticate_person(model, person_names::Vector{String}, img_path::String, 
                        expected_person::String; confidence_threshold::Float64=0.7)
            -> Tuple{Bool, Float64, String}

    Autentica se uma imagem corresponde a uma pessoa esperada.
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
    # MENU DE IDENTIFICAÇÃO
    # ============================================================================

    """
        show_identification_menu()

    Exibe menu interativo para escolher modo de identificação.
    """
    function show_identification_menu()
        println("\n" * "="^70)
        println("🎯 SISTEMA DE IDENTIFICAÇÃO FACIAL")
        println("="^70 * "\n")
        
        # Carregar modelo uma vez
        local model, person_names, config, model_metadata
        
        try
            model, person_names, config, model_metadata = load_model_for_inference()
        catch e
            @error "Erro ao carregar modelo" exception=(e, catch_backtrace())
            println("\n❌ Não foi possível carregar o modelo!")
            println("Certifique-se de que o modelo foi treinado primeiro.")
            return
        end
        
        Menu.run_menu([
            "📷 Identificar de arquivo de imagem",
            "🎥 Identificar de webcam (tempo real)",
            "📁 Identificação em lote (diretório)",
            "🔐 Autenticar pessoa",
            "ℹ️ Informações do modelo",
            "🔙 Voltar ao menu principal"
        ]; handlers=Dict(
            1 => () -> identify_from_file(model, person_names),
            2 => () -> identify_from_webcam(model, person_names),
            3 => () -> identify_batch_mode(model, person_names),
            4 => () -> authenticate_mode(model, person_names),
            5 => () -> show_model_info(config, model_metadata, person_names),
            6 => () -> println("🔙 Voltando ao menu principal...")
        ), loop=true)
    end

    """
        identify_from_file(model, person_names::Vector{String})

    Identifica pessoa a partir de um arquivo de imagem.
    """
    function identify_from_file(model, person_names::Vector{String})
        print("\n📸 Digite o caminho da imagem: ")
        flush(stdout)
        img_path = strip(readline())
        
        if isempty(img_path)
            println("❌ Caminho não fornecido")
            return
        end
        
        if !isfile(img_path)
            println("❌ Arquivo não encontrado: $img_path")
            return
        end
        
        try
            person_name, confidence = predict_person(model, person_names, img_path)
            
            if person_name !== nothing
                display_prediction_result(person_name, confidence, img_path)
            else
                println("❌ Falha na identificação da imagem")
            end
        catch e
            @error "Erro durante identificação" exception=(e, catch_backtrace())
            println("❌ Erro ao processar imagem")
        end
    end

    """
        identify_batch_mode(model, person_names::Vector{String})

    Identifica múltiplas imagens de um diretório.
    """
    function identify_batch_mode(model, person_names::Vector{String})
        print("\n📁 Digite o caminho do diretório com imagens: ")
        flush(stdout)
        dir_path = strip(readline())
        
        if isempty(dir_path)
            println("❌ Caminho não fornecido")
            return
        end
        
        if !isdir(dir_path)
            println("❌ Diretório não encontrado: $dir_path")
            return
        end
        
        print("💾 Nome do arquivo de resultados (Enter para padrão 'batch_results.txt'): ")
        flush(stdout)
        output_file = strip(readline())
        output_file = isempty(output_file) ? "batch_results.txt" : output_file
        
        try
            batch_identify(model, person_names, dir_path; output_file=output_file)
        catch e
            @error "Erro durante identificação em lote" exception=(e, catch_backtrace())
            println("❌ Erro ao processar diretório")
        end
    end

    """
        authenticate_mode(model, person_names::Vector{String})

    Autentica se uma imagem corresponde a uma pessoa esperada.
    """
    function authenticate_mode(model, person_names::Vector{String})
        println("\n🔐 MODO DE AUTENTICAÇÃO")
        println("="^50)
        println("Pessoas disponíveis:")
        for (i, name) in enumerate(person_names)
            println("  $i. $name")
        end
        
        print("\n👤 Digite o nome da pessoa esperada: ")
        flush(stdout)
        expected_person = strip(readline())
        
        if isempty(expected_person)
            println("❌ Nome não fornecido")
            return
        end
        
        if !(expected_person in person_names)
            println("⚠️ Aviso: Pessoa '$expected_person' não está no modelo treinado")
            print("Deseja continuar mesmo assim? (s/N): ")
            flush(stdout)
            response = strip(lowercase(readline()))
            if response != "s"
                return
            end
        end
        
        print("\n📸 Digite o caminho da imagem: ")
        flush(stdout)
        img_path = strip(readline())
        
        if isempty(img_path) || !isfile(img_path)
            println("❌ Arquivo não encontrado")
            return
        end
        
        print("🎚️ Limite de confiança (0.0-1.0, Enter para 0.7): ")
        flush(stdout)
        threshold_str = strip(readline())
        confidence_threshold = isempty(threshold_str) ? 0.7 : parse(Float64, threshold_str)
        
        try
            is_authenticated, confidence, status = authenticate_person(
                model, 
                person_names, 
                img_path, 
                expected_person;
                confidence_threshold=confidence_threshold
            )
            
            println("\n" * "="^70)
            println("🔍 RESULTADO DA AUTENTICAÇÃO")
            println("="^70)
            println("Esperado: $expected_person")
            println("Confiança: $(round(confidence*100, digits=2))%")
            println("Status: $status")
            println("="^70 * "\n")
            
        catch e
            @error "Erro durante autenticação" exception=(e, catch_backtrace())
            println("❌ Erro ao processar autenticação")
        end
    end

    """
        show_model_info(config::Dict, model_metadata, person_names::Vector{String})

    Exibe informações sobre o modelo carregado.
    """
    function show_model_info(config::Dict, model_metadata, person_names::Vector{String})
        println("\n" * "="^70)
        println("ℹ️ INFORMAÇÕES DO MODELO")
        println("="^70)
        
        println("\n📋 Configuração Geral:")
        println("  • Versão: $(config["metadata"]["version"])")
        println("  • Criado por: $(config["metadata"]["created_by"])")
        println("  • Último salvamento: $(config["metadata"]["last_saved"])")
        
        println("\n🎯 Modelo:")
        println("  • Arquitetura: $(config["model"]["model_architecture"])")
        println("  • Dimensões de entrada: $(config["model"]["img_width"])×$(config["model"]["img_height"])")
        println("  • Número de classes: $(config["model"]["num_classes"])")
        println("  • Augmentation usado: $(config["model"]["augmentation_used"])")
        
        println("\n📊 Treinamento:")
        println("  • Epochs treinadas: $(config["training"]["epochs_trained"])")
        println("  • Acurácia final: $(round(config["training"]["final_accuracy"]*100, digits=2))%")
        println("  • Melhor epoch: $(config["training"]["best_epoch"])")
        println("  • Learning rate: $(config["training"]["learning_rate"])")
        
        println("\n👥 Pessoas Reconhecidas ($(length(person_names))):")
        for (i, name) in enumerate(person_names)
            println("  $i. $name")
        end
        
        if haskey(config, "incremental_stats")
            println("\n🔄 Aprendizado Incremental:")
            println("  • Pessoas adicionadas: $(config["incremental_stats"]["new_people_added"])")
            println("  • Última atualização: $(config["incremental_stats"]["last_incremental_training"])")
        end
        
        println("\n" * "="^70 * "\n")
    end

    # ============================================================================
    # INTERFACE DE LINHA DE COMANDO
    # ============================================================================

    """
        main()

    Função principal - suporta tanto menu interativo quanto linha de comando.
    """
    function main()
        # Importar ARGS do Base
        args = Base.ARGS
        
        # Se chamado sem argumentos, mostrar menu interativo
        if length(args) == 0
            show_identification_menu()
            return
        end
        
        # Processar argumentos de linha de comando
        if args[1] == "--batch"
            if length(args) < 2
                @error "Especifique o diretório para identificação em lote"
                return
            end
            
            try
                model, person_names, _, _ = load_model_for_inference()
                output_file = length(args) >= 3 ? args[3] : "batch_results.txt"
                batch_identify(model, person_names, args[2]; output_file=output_file)
            catch e
                @error "Erro" exception=(e, catch_backtrace())
            end
            
        elseif length(args) >= 3 && args[2] == "--auth"
            # Modo autenticação
            img_path = args[1]
            expected_person = args[3]
            
            try
                model, person_names, _, _ = load_model_for_inference()
                is_authenticated, confidence, status = authenticate_person(
                    model, 
                    person_names,
                    img_path, 
                    expected_person
                )
                
                if is_authenticated
                    println("\n🎉 Autenticação bem-sucedida!")
                else
                    println("\n🚫 Autenticação falhou!")
                end
            catch e
                @error "Erro" exception=(e, catch_backtrace())
            end
            
        elseif args[1] == "--webcam"
            # Modo webcam via linha de comando
            try
                model, person_names, _, _ = load_model_for_inference()
                identify_from_webcam(model, person_names)
            catch e
                @error "Erro" exception=(e, catch_backtrace())
            end
            
        else
            # Modo identificação simples
            img_path = args[1]
            
            try
                model, person_names, _, _ = load_model_for_inference()
                person_name, confidence = predict_person(model, person_names, img_path)
                
                if person_name !== nothing
                    display_prediction_result(person_name, confidence, img_path)
                    println("✅ Identificação concluída com sucesso!")
                else
                    println("❌ Identificação não foi possível")
                end
            catch e
                @error "Erro" exception=(e, catch_backtrace())
            end
        end
    end

end # module Identif

# ============================================================================
# EXEMPLOS DE USO
# ============================================================================

# Identificação simples: julia cnncheckin_identify.jl foto.jpg
# Autenticação: julia cnncheckin_identify.jl foto.jpg --auth "Nome Pessoa"
# Lote: julia cnncheckin_identify.jl --batch ./diretorio/
# Webcam: julia cnncheckin_identify.jl --webcam

# Ou pelo menu interativo:
# julia cnncheckin_identify.jl

# ============================================================================
# EXECUÇÃO
# ============================================================================

 


# Identificação simples: julia cnncheckin_identify.jl foto.jpg
# Autenticação: julia cnncheckin_identify.jl foto.jpg --auth "Nome Pessoa"
# Lote: julia cnncheckin_identify.jl --batch ./diretorio/

# Identificação simples: julia cnncheckin_identify.jl ../../../dados/fotos_auth/nl.jpg

#  julia cnncheckin_identify.jl ../../../dados/fotos_auth/objeto-1.jpg
#  julia cnncheckin_identify.jl ../../../dados/fotos_auth/objeto-2.jpg
#  julia cnncheckin_identify.jl ../../../dados/fotos_auth/objeto-3.jpeg

#  julia cnncheckin_identify.jl ../../../dados/fotos_auth/teste.png