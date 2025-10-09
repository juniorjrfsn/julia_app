# projeto: cnncheckin
# file: cnncheckin/src/cnncheckin_incremental_webcam.jl
# descrição: Aprendizado incremental com captura via webcam

using Flux
using Statistics
using Random
using JLD2
using Dates

include("cnncheckin_core.jl")
include("cnncheckin_webcam.jl")

using .CNNCheckinCore
using .CNNCheckinWebcam

# Importar funções do incremental original
include("cnncheckin_incremental.jl")

# ============================================================================
# INTERFACE COM CAPTURA INTEGRADA
# ============================================================================

"""
    incremental_with_webcam_workflow()

Fluxo completo de aprendizado incremental com captura via webcam.
"""
function incremental_with_webcam_workflow()
    println("\n" * "="^70)
    println("📚 APRENDIZADO INCREMENTAL COM CAPTURA INTEGRADA")
    println("="^70)
    
    # Verificar modelo pré-treinado
    if !isfile(CNNCheckinCore.MODEL_PATH)
        println("\n❌ Modelo não encontrado!")
        println("\n💡 Primeiro execute o treinamento inicial:")
        println("   julia cnncheckin_pretrain_webcam.jl")
        return false
    end
    
    # Carregar configuração
    config = CNNCheckinCore.load_config(CNNCheckinCore.CONFIG_PATH)
    existing_people = config["data"]["person_names"]
    
    println("\n👥 Pessoas já treinadas no modelo:")
    for (i, person) in enumerate(existing_people)
        println("   $i. $person")
    end
    
    # Verificar imagens incrementais existentes
    incremental_dir = CNNCheckinCore.INCREMENTAL_DATA_PATH
    existing_images = if isdir(incremental_dir)
        filter(f -> lowercase(splitext(f)[2]) in CNNCheckinCore.VALID_IMAGE_EXTENSIONS,
               readdir(incremental_dir))
    else
        String[]
    end
    
    println("\n📊 Diretório incremental: $incremental_dir")
    println("   Imagens existentes: $(length(existing_images))")
    
    if !isempty(existing_images)
        new_people = Set{String}()
        for img in existing_images
            name = CNNCheckinCore.extract_person_name(img)
            if !(name in existing_people)
                push!(new_people, name)
            end
        end
        
        if !isempty(new_people)
            println("   Novas pessoas detectadas: $(join(sort(collect(new_people)), ", "))")
        end
    end
    
    # Menu de opções
    println("\n" * "─"^70)
    println("Escolha uma opção:")
    println("   1. Capturar nova pessoa via webcam")
    println("   2. Usar imagens existentes e treinar")
    println("   3. Adicionar mais pessoas E treinar")
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
        println("\n📸 Modo de captura para aprendizado incremental")
        println("═"^70)
        
        while true
            print("\nNome da NOVA pessoa (ou ENTER para finalizar): ")
            person_name = strip(readline())
            
            if isempty(person_name)
                break
            end
            
            # Verificar se pessoa já existe
            clean_name = replace(person_name, r"[^\w\s-]" => "")
            clean_name = replace(clean_name, " " => "_")
            
            if clean_name in existing_people
                println("\n⚠️  ATENÇÃO: '$clean_name' já está no modelo!")
                print("   Capturar mesmo assim? (s/N): ")
                if lowercase(strip(readline())) != "s"
                    continue
                end
            end
            
            print("Número de imagens (padrão 10): ")
            num_str = strip(readline())
            num_images = isempty(num_str) ? 10 : parse(Int, num_str)
            
            # Capturar imagens
            captured = CNNCheckinWebcam.capture_multiple_images(
                clean_name,
                incremental_dir,
                num_images;
                camera_index=CNNCheckinWebcam.get_recommended_camera(),
                delay_between=2
            )
            
            if captured < div(num_images * 3, 4)
                println("\n⚠️  Poucas imagens capturadas ($captured/$num_images)")
                print("Continuar mesmo assim? (s/N): ")
                if lowercase(strip(readline())) != "s"
                    continue
                end
            end
            
            print("\nCapturar outra pessoa? (s/N): ")
            if lowercase(strip(readline())) != "s"
                break
            end
        end
    end
    
    # Verificar se há imagens novas suficientes
    updated_images = filter(
        f -> lowercase(splitext(f)[2]) in CNNCheckinCore.VALID_IMAGE_EXTENSIONS,
        readdir(incremental_dir)
    )
    
    # Contar novas pessoas
    new_people_count = Dict{String, Int}()
    for img in updated_images
        name = CNNCheckinCore.extract_person_name(img)
        if !(name in existing_people)
            new_people_count[name] = get(new_people_count, name, 0) + 1
        end
    end
    
    if isempty(new_people_count)
        println("\n❌ Nenhuma pessoa nova encontrada!")
        println("\n💡 Dicas:")
        println("   - Capture imagens de pessoas diferentes das já treinadas")
        println("   - Use nomes diferentes: $(join(existing_people, ", "))")
        return false
    end
    
    # Filtrar pessoas com poucas imagens
    valid_people = filter(p -> p[2] >= 3, collect(new_people_count))
    
    if isempty(valid_people)
        println("\n❌ Nenhuma pessoa nova com imagens suficientes!")
        println("   Mínimo: 3 imagens por pessoa")
        for (name, count) in new_people_count
            println("   • $name: $count imagens ❌")
        end
        return false
    end
    
    # Confirmar treinamento incremental
    println("\n" * "═"^70)
    println("📊 Resumo do aprendizado incremental:")
    println("   Pessoas no modelo atual: $(length(existing_people))")
    println("   Novas pessoas válidas: $(length(valid_people))")
    
    for (name, count) in sort(valid_people)
        println("      • $name: $count imagens ✓")
    end
    
    invalid_people = filter(p -> p[2] < 3, collect(new_people_count))
    if !isempty(invalid_people)
        println("\n   ⚠️  Pessoas com poucas imagens (serão ignoradas):")
        for (name, count) in invalid_people
            println("      • $name: $count imagens")
        end
    end
    
    println("\n   Total após treinamento: $(length(existing_people) + length(valid_people)) pessoas")
    println("═"^70)
    print("\n🚀 Iniciar aprendizado incremental? (S/n): ")
    
    if lowercase(strip(readline())) == "n"
        println("❌ Treinamento cancelado")
        return false
    end
    
    # Executar aprendizado incremental
    println("\n🎯 Iniciando aprendizado incremental...")
    return incremental_learning_command()
end

# ============================================================================
# MODO RÁPIDO
# ============================================================================

"""
    quick_incremental(new_people::Vector{String}, images_per_person::Int=10)

Modo rápido: captura novas pessoas e treina incrementalmente.
"""
function quick_incremental(new_people::Vector{String}, images_per_person::Int=10)
    println("\n" * "="^70)
    println("⚡ MODO RÁPIDO: APRENDIZADO INCREMENTAL")
    println("="^70)
    
    # Verificar modelo
    if !isfile(CNNCheckinCore.MODEL_PATH)
        println("\n❌ Modelo não encontrado!")
        println("   Execute primeiro: julia cnncheckin_pretrain_webcam.jl")
        return false
    end
    
    # Carregar pessoas existentes
    config = CNNCheckinCore.load_config(CNNCheckinCore.CONFIG_PATH)
    existing_people = config["data"]["person_names"]
    
    println("\n📋 Pessoas a adicionar: $(join(new_people, ", "))")
    println("📸 Imagens por pessoa: $images_per_person")
    println("\n👥 Pessoas já no modelo: $(join(existing_people, ", "))")
    
    # Verificar conflitos
    conflicts = filter(p -> p in existing_people, new_people)
    if !isempty(conflicts)
        println("\n⚠️  ATENÇÃO: Pessoas já existem no modelo:")
        for person in conflicts
            println("   • $person")
        end
        print("\nContinuar mesmo assim? (s/N): ")
        if lowercase(strip(readline())) != "s"
            println("❌ Operação cancelada")
            return false
        end
    end
    
    incremental_dir = CNNCheckinCore.INCREMENTAL_DATA_PATH
    camera_index = CNNCheckinWebcam.get_recommended_camera()
    
    # Verificar câmera
    if !CNNCheckinWebcam.check_camera_available(camera_index)
        println("\n❌ Câmera não disponível!")
        return false
    end
    
    # Capturar todas as novas pessoas
    total_captured = 0
    successful_people = String[]
    
    for (i, person_name) in enumerate(new_people)
        println("\n" * "─"^70)
        println("[$i/$(length(new_people))] Capturando: $person_name")
        println("─"^70)
        
        print("⏸️  Pressione ENTER quando $person_name estiver pronto...")
        readline()
        
        captured = CNNCheckinWebcam.capture_multiple_images(
            person_name,
            incremental_dir,
            images_per_person;
            camera_index=camera_index,
            delay_between=2
        )
        
        total_captured += captured
        
        if captured >= max(3, div(images_per_person * 3, 4))
            push!(successful_people, person_name)
        else
            println("\n⚠️  Poucas imagens de $person_name ($captured/$images_per_person)")
        end
    end
    
    # Resumo
    println("\n" * "="^70)
    println("📊 RESUMO DA CAPTURA INCREMENTAL")
    println("="^70)
    println("   Pessoas solicitadas: $(length(new_people))")
    println("   Pessoas capturadas: $(length(successful_people))")
    println("   Total de imagens: $total_captured")
    
    if !isempty(successful_people)
        println("\n   ✅ Pessoas prontas para treinamento:")
        for person in successful_people
            println("      • $person")
        end
    end
    
    failed_people = filter(p -> !(p in successful_people), new_people)
    if !isempty(failed_people)
        println("\n   ⚠️  Pessoas com captura insuficiente:")
        for person in failed_people
            println("      • $person")
        end
    end
    
    if isempty(successful_people)
        println("\n❌ Nenhuma pessoa capturada com sucesso!")
        return false
    end
    
    print("\n🚀 Prosseguir com aprendizado incremental? (S/n): ")
    if lowercase(strip(readline())) == "n"
        println("❌ Treinamento cancelado")
        return false
    end
    
    # Executar aprendizado incremental
    return incremental_learning_command()
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
        # Modo interativo
        incremental_with_webcam_workflow()
        
    elseif ARGS[1] == "--quick" || ARGS[1] == "-q"
        # Modo rápido
        if length(ARGS) < 2
            println("""
            ❌ Uso: julia cnncheckin_incremental_webcam.jl --quick <pessoa1> <pessoa2> ... [--num N]
            
            Exemplo:
              julia cnncheckin_incremental_webcam.jl --quick "Carlos" "Ana" --num 10
            """)
            return
        end
        
        # Extrair nomes e número de imagens
        new_people = String[]
        images_per_person = 10
        
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
                push!(new_people, ARGS[i])
                i += 1
            end
        end
        
        if isempty(new_people)
            println("❌ Nenhuma pessoa especificada!")
            return
        end
        
        quick_incremental(new_people, images_per_person)
        
    elseif ARGS[1] == "--no-capture"
        # Treinar sem captura (usar imagens existentes)
        println("📚 Modo incremental sem captura")
        incremental_learning_command()
        
    elseif ARGS[1] == "--help" || ARGS[1] == "-h"
        println("""
        📚 CNNCheckin - Aprendizado Incremental com Webcam
        
        USO:
          julia cnncheckin_incremental_webcam.jl [opções]
        
        MODOS:
          (sem argumentos)              Modo interativo com menu
          --quick, -q <pessoas...>      Modo rápido (captura + treino incremental)
          --no-capture                  Treinar apenas com imagens existentes
          --help, -h                    Mostrar esta ajuda
        
        OPÇÕES DO MODO RÁPIDO:
          --num, -n <número>            Número de imagens por pessoa (padrão: 10)
        
        EXEMPLOS:
          # Modo interativo
          julia cnncheckin_incremental_webcam.jl
          
          # Modo rápido: adicionar 2 pessoas com 10 fotos cada
          julia cnncheckin_incremental_webcam.jl --quick "Carlos Alberto" "Ana Paula" --num 10
          
          # Apenas treinar (sem captura)
          julia cnncheckin_incremental_webcam.jl --no-capture
        
        FLUXO COMPLETO:
          1. Treinamento inicial:
             julia cnncheckin_pretrain_webcam.jl --quick "João" "Maria"
          
          2. Adicionar novas pessoas:
             julia cnncheckin_incremental_webcam.jl --quick "Carlos" "Ana"
          
          3. Identificar:
             julia cnncheckin_identify_webcam.jl
        
        REQUISITOS:
          - Modelo pré-treinado existente
          - Webcam conectada e funcional
          - Mínimo 3 imagens por nova pessoa
          - Nomes diferentes das pessoas já treinadas
        
        DICAS:
          - Capture pelo menos 8-10 imagens por pessoa
          - Varie poses e expressões
          - Use boa iluminação
          - Evite adicionar muitas pessoas de uma vez
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