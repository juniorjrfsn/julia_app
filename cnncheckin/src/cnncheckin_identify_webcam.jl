# projeto: cnncheckin
# file: cnncheckin/src/cnncheckin_identify_webcam.jl
# descrição: Identificação de pessoas com captura via webcam

using Flux
using JLD2
using Statistics
using Dates

include("cnncheckin_core.jl")
include("cnncheckin_webcam.jl")

using .CNNCheckinCore
using .CNNCheckinWebcam

# Importar funções do identify original
include("cnncheckin_identify.jl")

# ============================================================================
# IDENTIFICAÇÃO COM WEBCAM
# ============================================================================

"""
    identify_from_webcam(; camera_index::Int=0, save_image::Bool=true) 
        -> Tuple{Union{String, Nothing}, Float64, String}

Captura imagem da webcam e identifica a pessoa.
"""
function identify_from_webcam(; camera_index::Int=0, save_image::Bool=true)
    println("\n" * "="^70)
    println("🎯 IDENTIFICAÇÃO VIA WEBCAM")
    println("="^70)
    
    # Verificar modelo
    if !isfile(CNNCheckinCore.MODEL_PATH)
        println("\n❌ Modelo não encontrado!")
        println("   Execute primeiro: julia cnncheckin_pretrain_webcam.jl")
        return nothing, 0.0, "error"
    end
    
    # Verificar câmera
    if !CNNCheckinWebcam.check_camera_available(camera_index)
        println("\n❌ Câmera $camera_index não disponível!")
        
        cameras = CNNCheckinWebcam.list_available_cameras()
        if !isempty(cameras)
            println("\n💡 Câmeras disponíveis: $(join(cameras, ", "))")
            print("Usar câmera $(cameras[1])? (S/n): ")
            if lowercase(strip(readline())) != "n"
                camera_index = cameras[1]
            else
                return nothing, 0.0, "error"
            end
        else
            return nothing, 0.0, "error"
        end
    end
    
    # Gerar caminho temporário
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    temp_filename = "identificacao_$(timestamp).jpg"
    temp_path = joinpath(CNNCheckinCore.AUTH_DATA_PATH, temp_filename)
    
    # Criar diretório se necessário
    if !isdir(CNNCheckinCore.AUTH_DATA_PATH)
        mkpath(CNNCheckinCore.AUTH_DATA_PATH)
    end
    
    println("\n📸 Preparando para capturar imagem...")
    println("   Posicione-se em frente à câmera")
    
    # Capturar imagem
    success = CNNCheckinWebcam.capture_single_image(
        temp_path;
        camera_index=camera_index,
        show_preview=true,
        countdown=3
    )
    
    if !success
        println("\n❌ Falha na captura da imagem")
        return nothing, 0.0, "error"
    end
    
    println("\n🔍 Identificando pessoa...")
    
    # Identificar
    person_name, confidence, status = identify_command(temp_path)
    
    # Remover arquivo temporário se não for para salvar
    if !save_image && isfile(temp_path)
        try
            rm(temp_path)
            println("🗑️  Imagem temporária removida")
        catch
        end
    end
    
    return person_name, confidence, status
end

"""
    authenticate_from_webcam(expected_person::String; 
                            camera_index::Int=0,
                            confidence_threshold::Float64=0.7)
        -> Tuple{Bool, Float64, String}

Captura imagem da webcam e autentica se é a pessoa esperada.
"""
function authenticate_from_webcam(expected_person::String;
                                 camera_index::Int=0,
                                 confidence_threshold::Float64=0.7)
    println("\n" * "="^70)
    println("🔐 AUTENTICAÇÃO VIA WEBCAM")
    println("="^70)
    println("\n👤 Pessoa esperada: $expected_person")
    println("📊 Limiar de confiança: $(round(confidence_threshold*100, digits=0))%")
    
    # Verificar modelo
    if !isfile(CNNCheckinCore.MODEL_PATH)
        println("\n❌ Modelo não encontrado!")
        return false, 0.0, "error"
    end
    
    # Verificar câmera
    if !CNNCheckinWebcam.check_camera_available(camera_index)
        println("\n❌ Câmera não disponível!")
        return false, 0.0, "error"
    end
    
    # Gerar caminho temporário
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    temp_filename = "auth_$(expected_person)_$(timestamp).jpg"
    temp_path = joinpath(CNNCheckinCore.AUTH_DATA_PATH, temp_filename)
    
    if !isdir(CNNCheckinCore.AUTH_DATA_PATH)
        mkpath(CNNCheckinCore.AUTH_DATA_PATH)
    end
    
    println("\n📸 Capturando imagem para autenticação...")
    
    # Capturar
    success = CNNCheckinWebcam.capture_single_image(
        temp_path;
        camera_index=camera_index,
        show_preview=true,
        countdown=3
    )
    
    if !success
        println("\n❌ Falha na captura")
        return false, 0.0, "error"
    end
    
    println("\n🔍 Verificando identidade...")
    
    # Autenticar
    is_authenticated, confidence, status = identify_command(
        temp_path;
        auth_mode=true,
        expected_person=expected_person
    )
    
    # Exibir resultado
    println("\n" * "="^70)
    if is_authenticated
        println("✅ AUTENTICAÇÃO BEM-SUCEDIDA!")
        println("   Pessoa: $expected_person")
        println("   Confiança: $(round(confidence*100, digits=2))%")
    else
        println("❌ AUTENTICAÇÃO FALHOU!")
        println("   Status: $status")
        println("   Confiança: $(round(confidence*100, digits=2))%")
    end
    println("="^70)
    
    return is_authenticated, confidence, status
end

# ============================================================================
# MODO CONTÍNUO
# ============================================================================

"""
    continuous_identification(; camera_index::Int=0, 
                             interval::Int=5,
                             max_attempts::Int=0)

Modo contínuo: identifica pessoas repetidamente.
"""
function continuous_identification(; camera_index::Int=0,
                                   interval::Int=5,
                                   max_attempts::Int=0)
    println("\n" * "="^70)
    println("🔄 MODO DE IDENTIFICAÇÃO CONTÍNUA")
    println("="^70)
    println("\n⚙️  Configurações:")
    println("   Câmera: $camera_index")
    println("   Intervalo: $interval segundos")
    println("   Tentativas: $(max_attempts == 0 ? "ilimitadas" : string(max_attempts))")
    
    # Verificar modelo
    if !isfile(CNNCheckinCore.MODEL_PATH)
        println("\n❌ Modelo não encontrado!")
        return false
    end
    
    # Verificar câmera
    if !CNNCheckinWebcam.check_camera_available(camera_index)
        println("\n❌ Câmera não disponível!")
        return false
    end
    
    # Carregar modelo uma vez
    println("\n📂 Carregando modelo...")
    model, person_names, config, model_metadata = load_model_for_inference()
    
    println("\n✅ Modelo carregado!")
    println("   Pessoas reconhecidas: $(join(person_names, ", "))")
    println("\n🎬 Iniciando identificação contínua...")
    println("   Pressione Ctrl+C para parar\n")
    
    attempt = 0
    results_log = []
    
    try
        while true
            attempt += 1
            
            if max_attempts > 0 && attempt > max_attempts
                println("\n✅ Número máximo de tentativas atingido")
                break
            end
            
            println("─"^70)
            println("[$attempt] $(Dates.format(now(), "HH:MM:SS"))")
            
            # Capturar imagem
            timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
            temp_path = joinpath(CNNCheckinCore.AUTH_DATA_PATH, "continuous_$(timestamp).jpg")
            
            success = CNNCheckinWebcam.capture_single_image(
                temp_path;
                camera_index=camera_index,
                show_preview=false,
                countdown=0
            )
            
            if success
                # Identificar
                person_name, confidence = predict_person(model, person_names, temp_path; save_example=false)
                
                if person_name !== nothing
                    # Emoji baseado na confiança
                    emoji = if confidence >= 0.9
                        "✅"
                    elseif confidence >= 0.7
                        "⚡"
                    else
                        "⚠️"
                    end
                    
                    println("$emoji Identificado: $person_name ($(round(confidence*100, digits=1))%)")
                    
                    # Registrar resultado
                    push!(results_log, Dict(
                        "attempt" => attempt,
                        "timestamp" => timestamp,
                        "person" => person_name,
                        "confidence" => confidence
                    ))
                else
                    println("❌ Falha na identificação")
                end
                
                # Remover imagem temporária
                try
                    rm(temp_path)
                catch
                end
            else
                println("❌ Falha na captura")
            end
            
            # Aguardar próxima tentativa
            if max_attempts == 0 || attempt < max_attempts
                print("⏳ Aguardando $interval segundos... ")
                flush(stdout)
                sleep(interval)
                println("✓")
            end
        end
        
    catch e
        if isa(e, InterruptException)
            println("\n\n⏹️  Identificação contínua interrompida pelo usuário")
        else
            println("\n\n❌ Erro: $e")
        end
    end
    
    # Resumo
    if !isempty(results_log)
        println("\n" * "="^70)
        println("📊 RESUMO DA SESSÃO CONTÍNUA")
        println("="^70)
        println("   Total de tentativas: $attempt")
        println("   Identificações bem-sucedidas: $(length(results_log))")
        println("   Taxa de sucesso: $(round(length(results_log)/attempt*100, digits=1))%")
        
        # Contar identificações por pessoa
        person_counts = Dict{String, Int}()
        for result in results_log
            person = result["person"]
            person_counts[person] = get(person_counts, person, 0) + 1
        end
        
        println("\n   Pessoas identificadas:")
        for (person, count) in sort(collect(person_counts), by=x->x[2], rev=true)
            println("      • $person: $count vezes")
        end
        
        # Confiança média
        avg_confidence = mean(r["confidence"] for r in results_log)
        println("\n   Confiança média: $(round(avg_confidence*100, digits=1))%")
        println("="^70)
    end
    
    return true
end

# ============================================================================
# MODO CHECK-IN/CHECK-OUT
# ============================================================================

"""
    checkin_system(; camera_index::Int=0, log_file::String="checkin_log.txt")

Sistema de check-in: registra entrada/saída de pessoas.
"""
function checkin_system(; camera_index::Int=0, log_file::String="checkin_log.txt")
    println("\n" * "="^70)
    println("📋 SISTEMA DE CHECK-IN/CHECK-OUT")
    println("="^70)
    
    # Verificar modelo
    if !isfile(CNNCheckinCore.MODEL_PATH)
        println("\n❌ Modelo não encontrado!")
        return false
    end
    
    # Verificar câmera
    if !CNNCheckinWebcam.check_camera_available(camera_index)
        println("\n❌ Câmera não disponível!")
        return false
    end
    
    # Carregar modelo
    println("\n📂 Carregando modelo...")
    model, person_names, config, model_metadata = load_model_for_inference()
    
    println("\n✅ Sistema pronto!")
    println("   Pessoas cadastradas: $(join(person_names, ", "))")
    println("   Log: $log_file")
    
    # Registro de presença
    present_people = Set{String}()
    
    println("\n🎬 Sistema de check-in ativo")
    println("   Digite 'sair' para encerrar\n")
    
    attempt = 0
    
    while true
        attempt += 1
        
        println("\n" * "─"^70)
        println("⏸️  Pressione ENTER para registrar check-in (ou 'sair' para encerrar):")
        response = readline()
        
        if lowercase(strip(response)) == "sair"
            break
        end
        
        println("\n📸 Capturando...")
        
        # Capturar imagem
        timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
        temp_path = joinpath(CNNCheckinCore.AUTH_DATA_PATH, "checkin_$(timestamp).jpg")
        
        success = CNNCheckinWebcam.capture_single_image(
            temp_path;
            camera_index=camera_index,
            show_preview=true,
            countdown=3
        )
        
        if !success
            println("❌ Falha na captura")
            continue
        end
        
        # Identificar
        println("🔍 Identificando...")
        person_name, confidence = predict_person(model, person_names, temp_path; save_example=false)
        
        if person_name === nothing || confidence < 0.6
            println("❌ Pessoa não identificada ou confiança baixa")
            println("   Tente novamente")
            try
                rm(temp_path)
            catch
            end
            continue
        end
        
        # Determinar check-in ou check-out
        is_checkin = !(person_name in present_people)
        
        if is_checkin
            push!(present_people, person_name)
            action = "CHECK-IN"
            emoji = "✅"
        else
            delete!(present_people, person_name)
            action = "CHECK-OUT"
            emoji = "👋"
        end
        
        # Exibir resultado
        println("\n" * "═"^70)
        println("$emoji $action REGISTRADO!")
        println("   Pessoa: $person_name")
        println("   Confiança: $(round(confidence*100, digits=1))%")
        println("   Data/Hora: $(Dates.format(now(), "dd/mm/yyyy HH:MM:SS"))")
        println("═"^70)
        
        # Registrar em arquivo
        try
            open(log_file, "a") do io
                println(io, "$(Dates.format(now(), "yyyy-mm-dd HH:MM:SS")),$action,$person_name,$(round(confidence, digits=4))")
            end
            println("💾 Registro salvo no log")
        catch e
            println("⚠️  Erro ao salvar log: $e")
        end
        
        # Mostrar quem está presente
        if !isempty(present_people)
            println("\n👥 Pessoas presentes: $(join(sort(collect(present_people)), ", "))")
        else
            println("\n👥 Nenhuma pessoa presente no momento")
        end
        
        # Limpar imagem temporária
        try
            rm(temp_path)
        catch
        end
    end
    
    # Resumo final
    println("\n" * "="^70)
    println("📊 SESSÃO ENCERRADA")
    println("="^70)
    println("   Total de registros: $attempt")
    
    if !isempty(present_people)
        println("\n   ⚠️  Pessoas que não fizeram check-out:")
        for person in sort(collect(present_people))
            println("      • $person")
        end
    end
    
    println("\n   📄 Log completo: $log_file")
    println("="^70)
    
    return true
end

# ============================================================================
# INTERFACE DE LINHA DE COMANDO
# ============================================================================

"""
    main()

Função principal com suporte a webcam.
"""
function main()
    if length(ARGS) == 0
        # Menu interativo
        println("\n" * "="^70)
        println("🎯 SISTEMA DE IDENTIFICAÇÃO VIA WEBCAM")
        println("="^70)
        println("\nEscolha um modo:")
        println("   1. Identificação única")
        println("   2. Autenticação (verificar pessoa específica)")
        println("   3. Identificação contínua")
        println("   4. Sistema de check-in/check-out")
        println("   0. Sair")
        print("\nOpção: ")
        
        option = readline()
        
        if option == "1"
            identify_from_webcam()
            
        elseif option == "2"
            print("\nNome da pessoa esperada: ")
            expected = readline()
            print("Limiar de confiança (0.0-1.0, padrão 0.7): ")
            threshold_str = strip(readline())
            threshold = isempty(threshold_str) ? 0.7 : parse(Float64, threshold_str)
            authenticate_from_webcam(expected; confidence_threshold=threshold)
            
        elseif option == "3"
            print("\nIntervalo entre capturas (segundos, padrão 5): ")
            interval_str = strip(readline())
            interval = isempty(interval_str) ? 5 : parse(Int, interval_str)
            
            print("Número máximo de tentativas (0 = ilimitado): ")
            max_str = strip(readline())
            max_attempts = isempty(max_str) ? 0 : parse(Int, max_str)
            
            continuous_identification(interval=interval, max_attempts=max_attempts)
            
        elseif option == "4"
            print("\nArquivo de log (padrão checkin_log.txt): ")
            log_file = strip(readline())
            log_file = isempty(log_file) ? "checkin_log.txt" : log_file
            checkin_system(log_file=log_file)
            
        elseif option == "0"
            println("👋 Até logo!")
        else
            println("❌ Opção inválida")
        end
        
    elseif ARGS[1] == "--identify" || ARGS[1] == "-i"
        camera = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 0
        identify_from_webcam(camera_index=camera)
        
    elseif ARGS[1] == "--auth" || ARGS[1] == "-a"
        if length(ARGS) < 2
            println("❌ Uso: julia cnncheckin_identify_webcam.jl --auth <pessoa> [threshold]")
            return
        end
        
        expected = ARGS[2]
        threshold = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.7
        authenticate_from_webcam(expected; confidence_threshold=threshold)
        
    elseif ARGS[1] == "--continuous" || ARGS[1] == "-c"
        interval = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 5
        max_attempts = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 0
        continuous_identification(interval=interval, max_attempts=max_attempts)
        
    elseif ARGS[1] == "--checkin"
        log_file = length(ARGS) >= 2 ? ARGS[2] : "checkin_log.txt"
        checkin_system(log_file=log_file)
        
    elseif ARGS[1] == "--help" || ARGS[1] == "-h"
        println("""
        🎯 CNNCheckin - Identificação com Webcam
        
        USO:
          julia cnncheckin_identify_webcam.jl [comando] [argumentos]
        
        COMANDOS:
          (sem argumentos)                     Menu interativo
          --identify, -i [camera]              Identificação única
          --auth, -a <pessoa> [threshold]      Autenticação
          --continuous, -c [intervalo] [max]   Modo contínuo
          --checkin [arquivo_log]              Sistema check-in/check-out
          --help, -h                           Mostrar ajuda
        
        EXEMPLOS:
          # Menu interativo
          julia cnncheckin_identify_webcam.jl
          
          # Identificação única
          julia cnncheckin_identify_webcam.jl --identify
          
          # Autenticar pessoa específica
          julia cnncheckin_identify_webcam.jl --auth "João Silva" 0.75
          
          # Modo contínuo (captura a cada 10 segundos)
          julia cnncheckin_identify_webcam.jl --continuous 10
          
          # Sistema de check-in
          julia cnncheckin_identify_webcam.jl --checkin presenca.csv
        
        MODOS DE USO:
        
          1. IDENTIFICAÇÃO ÚNICA
             - Captura uma imagem
             - Identifica a pessoa
             - Mostra resultado
          
          2. AUTENTICAÇÃO
             - Verifica se a pessoa é quem diz ser
             - Útil para controle de acesso
          
          3. CONTÍNUO
             - Identifica pessoas repetidamente
             - Útil para monitoramento
          
          4. CHECK-IN/CHECK-OUT
             - Registra entrada e saída
             - Mantém log de presença
             - Gera arquivo CSV
        """)
        
    else
        println("❌ Comando desconhecido: $(ARGS[1])")
        println("Use --help para ver os comandos disponíveis")
    end
end

# ============================================================================
# EXECUÇÃO
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end