# projeto: cnncheckin
# file: cnncheckin/src/cnncheckin_pretrain_webcam.jl
# descrição: Script de pré-treinamento com opção de captura via webcam

using Flux
using Statistics
using Random
using JLD2
using Dates

include("cnncheckin_core.jl")
include("cnncheckin_webcam.jl")

using .CNNCheckinCore
using .CNNCheckinWebcam

# Importar funções do pretrain original
include("cnncheckin_pretrain.jl")

# ============================================================================
# INTERFACE COM CAPTURA INTEGRADA
# ============================================================================

"""
    pretrain_with_webcam_workflow()

Fluxo completo de treinamento com opção de captura via webcam.
"""
function pretrain_with_webcam_workflow()
    println("\n" * "="^70)
    println("🎓 SISTEMA DE TREINAMENTO COM CAPTURA INTEGRADA")
    println("="^70)
    
    # Verificar imagens existentes
    train_dir = CNNCheckinCore.TRAIN_DATA_PATH
    existing_images = if isdir(train_dir)
        filter(f -> lowercase(splitext(f)[2]) in CNNCheckinCore.VALID_IMAGE_EXTENSIONS, 
               readdir(train_dir))
    else
        String[]
    end
    
    println("\n📊 Status atual:")
    println("   Diretório: $train_dir")
    println("   Imagens existentes: $(length(existing_images))")
    
    if !isempty(existing_images)
        # Contar pessoas
        people = Set{String}()
        for img in existing_images
            push!(people, CNNCheckinCore.extract_person_name(img))
        end
        println("   Pessoas detectadas: $(length(people))")
        println("   → $(join(sort(collect(people)), ", "))")
    end
    
    # Menu de opções
    println("\n" * "─"^70)
    println("Escolha uma opção:")
    println("   1. Capturar novas imagens via webcam")
    println("   2. Usar imagens existentes e treinar")
    println("   3. Adicionar mais imagens E treinar")
    println("   0. Cancelar")
    println("─"^70)
    print("\nOpção: ")
    
    option = readline()
    
    if option == "0"
        println("❌ Operação cancelada")
        return false
    end
    
    # Capturar novas imagens se necessário
    if option == "1" || option == "3"
        println("\n📸 Modo de captura de imagens")
        println("═"^70)
        
        while true
            print("\nNome da pessoa (ou ENTER para finalizar): ")
            person_name = strip(readline())
            
            if isempty(person_name)
                break
            end
            
            print("Número de imagens (padrão 15): ")
            num_str = strip(readline())
            num_images = isempty(num_str) ? 15 : parse(Int, num_str)
            
            # Capturar imagens
            success = CNNCheckinWebcam.capture_training_session(
                person_name,
                train_dir;
                num_images=num_images,
                camera_index=CNNCheckinWebcam.get_recommended_camera()
            )
            
            if !success
                println("\n⚠️  Captura não foi totalmente bem-sucedida")
                print("Continuar com próxima pessoa? (s/N): ")
                if lowercase(strip(readline())) != "s"
                    break
                end
            end
            
            print("\nCapturar outra pessoa? (S/n): ")
            if lowercase(strip(readline())) == "n"
                break
            end
        end
    end
    
    # Verificar se há imagens suficientes
    updated_images = filter(
        f -> lowercase(splitext(f)[2]) in CNNCheckinCore.VALID_IMAGE_EXTENSIONS,
        readdir(train_dir)
    )
    
    if length(updated_images) < 10
        println("\n❌ Imagens insuficientes para treinamento!")
        println("   Mínimo recomendado: 10 imagens (3+ por pessoa)")
        println("   Encontradas: $(length(updated_images))")
        return false
    end
    
    # Confirmar treinamento
    println("\n" * "═"^70)
    println("📊 Resumo pré-treinamento:")
    println("   Total de imagens: $(length(updated_images))")
    
    # Contar pessoas
    people = Dict{String, Int}()
    for img in updated_images
        name = CNNCheckinCore.extract_person_name(img)
        people[name] = get(people, name, 0) + 1
    end
    
    println("   Pessoas encontradas: $(length(people))")
    for (name, count) in sort(collect(people))
        println("      • $name: $count imagens")
    end
    
    println("═"^70)
    print("\n🚀 Iniciar treinamento? (S/n): ")
    
    if lowercase(strip(readline())) == "n"
        println("❌ Treinamento cancelado")
        return false
    end
    
    # Executar treinamento
    println("\n🎯 Iniciando treinamento do modelo...")
    return pretrain_command()
end

# ============================================================================
# MODO RÁPIDO
# ============================================================================

"""
    quick_capture_and_train(people_names::Vector{String}, images_per_person::Int=15)

Modo rápido: captura múltiplas pessoas sequencialmente e treina.
"""
function quick_capture_and_train(people_names::Vector{String}, images_per_person::Int=15)
    println("\n" * "="^70)
    println("⚡ MODO RÁPIDO: CAPTURA E TREINAMENTO")
    println("="^70)
    println("\n📋 Pessoas a capturar: $(join(people_names, ", "))")
    println("📸 Imagens por pessoa: $images_per_person")
    
    train_dir = CNNCheckinCore.TRAIN_DATA_PATH
    camera_index = CNNCheckinWebcam.get_recommended_camera()
    
    # Verificar câmera
    if !CNNCheckinWebcam.check_camera_available(camera_index)
        println("\n❌ Câmera não disponível!")
        return false
    end
    
    # Capturar todas as pessoas
    total_captured = 0
    successful_people = String[]
    
    for (i, person_name) in enumerate(people_names)
        println("\n" * "─"^70)
        println("[$i/$(length(people_names))] Capturando: $person_name")
        println("─"^70)
        
        print("⏸️  Pressione ENTER quando $person_name estiver pronto...")
        readline()
        
        captured = CNNCheckinWebcam.capture_multiple_images(
            person_name,
            train_dir,
            images_per_person;
            camera_index=camera_index,
            delay_between=2
        )
        
        total_captured += captured
        
        if captured >= div(images_per_person * 3, 4)
            push!(successful_people, person_name)
        else
            println("\n⚠️  Poucas imagens de $person_name ($captured/$images_per_person)")
        end
    end
    
    # Resumo da captura
    println("\n" * "="^70)
    println("📊 RESUMO DA CAPTURA")
    println("="^70)
    println("   Pessoas solicitadas: $(length(people_names))")
    println("   Pessoas capturadas: $(length(successful_people))")
    println("   Total de imagens: $total_captured")
    println("   Média por pessoa: $(round(total_captured/length(people_names), digits=1))")
    
    if length(successful_people) < 2
        println("\n❌ Poucas pessoas capturadas para treinamento!")
        return false
    end
    
    print("\n🚀 Prosseguir com treinamento? (S/n): ")
    if lowercase(strip(readline())) == "n"
        println("❌ Treinamento cancelado")
        return false
    end
    
    # Executar treinamento
    return pretrain_command()
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
        # Modo interativo com webcam
        pretrain_with_webcam_workflow()
        
    elseif ARGS[1] == "--quick" || ARGS[1] == "-q"
        # Modo rápido
        if length(ARGS) < 2
            println("""
            ❌ Uso: julia cnncheckin_pretrain_webcam.jl --quick <pessoa1> <pessoa2> ... [--num N]
            
            Exemplo:
              julia cnncheckin_pretrain_webcam.jl --quick "João" "Maria" "Pedro" --num 12
            """)
            return
        end
        
        # Extrair nomes e número de imagens
        people_names = String[]
        images_per_person = 15
        
        i = 2
        while i <= length(ARGS)
            if ARGS[i] == "--num" || ARGS[i] == "-n"
                if i < length(ARGS)
                    images_per_person = parse(Int, ARGS[i+1])
                    i += 2
                else
                    i += 1
                end
            else
                push!(people_names, ARGS[i])
                i += 1
            end
        end
        
        if isempty(people_names)
            println("❌ Nenhuma pessoa especificada!")
            return
        end
        
        quick_capture_and_train(people_names, images_per_person)
        
    elseif ARGS[1] == "--no-capture"
        # Treinar sem captura (usar imagens existentes)
        println("🎓 Modo de treinamento sem captura")
        pretrain_command()
        
    elseif ARGS[1] == "--help" || ARGS[1] == "-h"
        println("""
        🎓 CNNCheckin - Treinamento com Captura via Webcam
        
        USO:
          julia cnncheckin_pretrain_webcam.jl [opções]
        
        MODOS:
          (sem argumentos)              Modo interativo com menu
          --quick, -q <pessoas...>      Modo rápido (captura sequencial + treino)
          --no-capture                  Treinar apenas com imagens existentes
          --help, -h                    Mostrar esta ajuda
        
        OPÇÕES DO MODO RÁPIDO:
          --num, -n <número>            Número de imagens por pessoa (padrão: 15)
        
        EXEMPLOS:
          # Modo interativo
          julia cnncheckin_pretrain_webcam.jl
          
          # Modo rápido: capturar 3 pessoas com 12 fotos cada
          julia cnncheckin_pretrain_webcam.jl --quick "João Silva" "Maria Santos" "Pedro Costa" --num 12
          
          # Apenas treinar (sem captura)
          julia cnncheckin_pretrain_webcam.jl --no-capture
        
        FLUXO RECOMENDADO:
          1. Execute este script no modo interativo ou rápido
          2. Siga as instruções de captura na tela
          3. O treinamento iniciará automaticamente
          4. Arquivos gerados:
             - face_recognition_model.jld2
             - face_recognition_config.toml
             - face_recognition_model_data.toml
        
        REQUISITOS:
          - Webcam conectada e funcional
          - Boa iluminação
          - Fundo neutro (recomendado)
          - Mínimo 10 imagens total (3+ por pessoa)
        """)
        
    else
        println("❌ Opção desconhecida: $(ARGS[1])")
        println("Use --help para ver as opções disponíveis")
    end
end

# ============================================================================
# EXECUÇÃO
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end