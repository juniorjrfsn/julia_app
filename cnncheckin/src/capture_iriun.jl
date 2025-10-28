#!/usr/bin/env julia
# Captura específica para Iriun Webcam

# projeto: cnncheckin
# file: cnncheckin/src/capture_iriun.jl

# Captura específica para Iriun Webcam
# Compatível com celulares Android/iOS via Iriun

using VideoIO
using Images
using FileIO
using Dates

"""
    detect_iriun_camera() -> Union{Int, Nothing}

Detecta automaticamente o índice da câmera Iriun.
"""
function detect_iriun_camera()
    println("🔍 Detectando Iriun Webcam...")
    
    # Linux: Iriun geralmente aparece como /dev/video2 ou superior
    # Testar múltiplos índices
    for camera_idx in 0:10
        try
            # Tentar abrir câmera
            cam = VideoIO.opencamera(camera_idx)
            
            if cam !== nothing
                # Testar captura
                frame = read(cam)
                close(cam)
                
                if frame !== nothing
                    height, width = size(frame)[1:2]
                    push!(available, i)
                    println("  ✔ Câmera $i: $(width)x$(height)")
                end
            end
        catch e
            continue
        end
    end
    
    if isempty(available)
        println("  ⚠️  Nenhuma câmera detectada")
        println("\n💡 Soluções:")
        println("   1. Verifique se Iriun Webcam está rodando no celular")
        println("   2. Conecte via USB ou WiFi (mesma rede)")
        println("   3. Reinicie o serviço: sudo systemctl restart iriunwebcam")
        println("   4. Verifique permissões: ls -l /dev/video*")
    else
        println("\n✅ Total de câmeras encontradas: $(length(available))")
        println("💡 Para Iriun, tente os índices maiores (geralmente 2+)")
    end
    
    return available
end

"""
    open_camera_with_retry(camera_index::Int; max_attempts::Int=3) -> Union{VideoIO.VideoReader, Nothing}

Abre câmera com múltiplas tentativas.
"""
function open_camera_with_retry(camera_index::Int; max_attempts::Int=3)
    for attempt in 1:max_attempts
        try
            println("   Tentativa $attempt/$max_attempts...")
            cam = VideoIO.opencamera(camera_index)
            
            if cam !== nothing
                # Testar se consegue ler frame
                frame = read(cam)
                if frame !== nothing
                    println("   ✅ Câmera $camera_index aberta com sucesso!")
                    return cam
                else
                    close(cam)
                end
            end
        catch e
            println("   ⚠️  Erro: $(typeof(e).name)")
            if attempt < max_attempts
                println("   ⏳ Aguardando 2 segundos...")
                sleep(2)
            end
        end
    end
    
    return nothing
end

"""
    capture_single_image_iriun(output_path::String; camera_index::Union{Int, Nothing}=nothing, countdown::Int=3) -> Bool

Captura uma única imagem usando Iriun Webcam.
"""
function capture_single_image_iriun(output_path::String; 
                                    camera_index::Union{Int, Nothing}=nothing, 
                                    countdown::Int=3)
    println("\n📸 Captura com Iriun Webcam")
    println("="^60)
    
    # Criar diretório se necessário
    output_dir = dirname(output_path)
    if !isempty(output_dir) && !isdir(output_dir)
        mkpath(output_dir)
    end
    
    # Detectar câmera se não especificado
    if camera_index === nothing
        camera_index = detect_iriun_camera()
        
        if camera_index === nothing
            println("\n⚠️  Câmera não detectada automaticamente")
            cameras = list_all_cameras()
            
            if !isempty(cameras)
                println("\n❓ Qual câmera deseja usar?")
                for cam in cameras
                    println("   [$cam] Câmera $cam")
                end
                print("\nÍndice da câmera (ou ENTER para $(cameras[1])): ")
                response = readline()
                camera_index = isempty(strip(response)) ? cameras[1] : parse(Int, strip(response))
            else
                println("\n❌ Nenhuma câmera disponível!")
                return false
            end
        end
    end
    
    println("\n🎥 Usando câmera: $camera_index")
    
    # Abrir câmera com retry
    cam = open_camera_with_retry(camera_index)
    
    if cam === nothing
        println("\n❌ Não foi possível abrir a câmera $camera_index")
        println("\n💡 Dicas:")
        println("   1. Verifique se Iriun está rodando: ps aux | grep iriun")
        println("   2. Reinicie o serviço: sudo systemctl restart iriunwebcam")
        println("   3. Teste com outro índice: julia capture_iriun.jl --test")
        return false
    end
    
    # Countdown
    if countdown > 0
        println("\n⏱️  Preparando captura em $countdown segundos...")
        println("   Posicione-se em frente à câmera...")
        
        for i in countdown:-1:1
            println("   $i...")
            sleep(1)
            
            # Ler frames durante countdown para "esquentar" câmera
            try
                read(cam)
            catch
            end
        end
    end
    
    # Capturar
    println("📸 Capturando...")
    
    try
        frame = read(cam)
        
        if frame !== nothing
            img = RGB.(frame)
            save(output_path, img)
            println("✅ Imagem salva: $output_path")
            
            # Mostrar info da imagem
            height, width = size(img)
            println("   Resolução: $(width)x$(height)")
            
            close(cam)
            return true
        else
            println("❌ Falha ao capturar frame")
            close(cam)
            return false
        end
        
    catch e
        println("❌ Erro durante captura: $e")
        close(cam)
        return false
    end
end

"""
    capture_multiple_images_iriun(person_name::String, output_dir::String, 
                                   num_images::Int=15; camera_index::Union{Int, Nothing}=nothing,
                                   delay_between::Int=2) -> Int

Captura múltiplas imagens de uma pessoa usando Iriun.
"""
function capture_multiple_images_iriun(person_name::String, 
                                       output_dir::String, 
                                       num_images::Int=15;
                                       camera_index::Union{Int, Nothing}=nothing,
                                       delay_between::Int=2)
    println("\n📸 Captura múltipla com Iriun Webcam")
    println("="^60)
    println("   Pessoa: $person_name")
    println("   Quantidade: $num_images imagens")
    println("   Intervalo: $delay_between segundos")
    println("   Destino: $output_dir")
    println("="^60)
    
    # Criar diretório
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    
    # Detectar câmera
    if camera_index === nothing
        camera_index = detect_iriun_camera()
        
        if camera_index === nothing
            cameras = list_all_cameras()
            if !isempty(cameras)
                camera_index = cameras[end]  # Usar última (geralmente Iriun)
                println("\n📱 Usando câmera $camera_index (última detectada)")
            else
                println("\n❌ Nenhuma câmera disponível!")
                return 0
            end
        end
    end
    
    # Abrir câmera
    cam = open_camera_with_retry(camera_index)
    
    if cam === nothing
        println("\n❌ Não foi possível abrir câmera")
        return 0
    end
    
    println("\n✅ Câmera inicializada")
    println("\n🎬 Iniciando sequência de capturas...")
    println("💡 Dica: Varie a posição e expressão entre as capturas\n")
    
    captured = 0
    
    for i in 1:num_images
        try
            println("[$i/$num_images] Preparando captura...")
            
            # Countdown
            for j in delay_between:-1:1
                print("   $j... ")
                flush(stdout)
                
                # Ler frame durante countdown
                try
                    read(cam)
                catch
                end
                
                sleep(1)
            end
            println("📸")
            
            # Capturar
            frame = read(cam)
            
            if frame === nothing
                println("   ⚠️  Falha ao capturar frame $i")
                continue
            end
            
            # Salvar
            timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
            filename = "$(person_name)-$(i)_$(timestamp).jpg"
            filepath = joinpath(output_dir, filename)
            
            img = RGB.(frame)
            save(filepath, img)
            
            captured += 1
            println("   ✅ Salva: $filename")
            
        catch e
            println("   ❌ Erro na captura $i: $e")
        end
    end
    
    # Fechar câmera
    try
        close(cam)
    catch
    end
    
    # Resumo
    println("\n" * "="^60)
    println("✅ Sessão finalizada!")
    println("   Imagens capturadas: $captured/$num_images")
    println("   Taxa de sucesso: $(round(captured/num_images*100, digits=1))%")
    
    if captured >= div(num_images * 3, 4)
        println("   🎉 Quantidade suficiente para treinamento!")
    else
        println("   ⚠️  Poucas imagens capturadas. Recomenda-se repetir.")
    end
    
    println("="^60)
    
    return captured
end

"""
    test_cameras()

Testa todas as câmeras disponíveis.
"""
function test_cameras()
    println("\n🧪 TESTE DE CÂMERAS")
    println("="^60)
    
    cameras = list_all_cameras()
    
    if isempty(cameras)
        return false
    end
    
    println("\n📹 Testando cada câmera com captura real...\n")
    
    for cam_idx in cameras
        println("─"^60)
        println("Testando câmera $cam_idx...")
        
        try
            cam = VideoIO.opencamera(cam_idx)
            
            if cam !== nothing
                # Capturar alguns frames
                frames_captured = 0
                
                for _ in 1:5
                    frame = read(cam)
                    if frame !== nothing
                        frames_captured += 1
                    end
                    sleep(0.2)
                end
                
                close(cam)
                
                if frames_captured >= 3
                    println("✅ Câmera $cam_idx: FUNCIONANDO ($frames_captured/5 frames)")
                else
                    println("⚠️  Câmera $cam_idx: INSTÁVEL ($frames_captured/5 frames)")
                end
            else
                println("❌ Câmera $cam_idx: Não abriu")
            end
            
        catch e
            println("❌ Câmera $cam_idx: Erro - $(typeof(e).name)")
        end
    end
    
    println("\n" * "="^60)
    
    # Detectar Iriun
    iriun_idx = detect_iriun_camera()
    if iriun_idx !== nothing
        println("\n🎯 Recomendação: Use a câmera $iriun_idx para Iriun Webcam")
    end
    
    return true
end

# ============================================================================
# INTERFACE DE LINHA DE COMANDO
# ============================================================================

"""
    main()

Função principal CLI.
"""
function main()
    if length(ARGS) == 0
        println("""
        USO:
          julia capture_iriun.jl --single <output.jpg> [--camera N]
          julia capture_iriun.jl --multiple <nome> <diretorio> <num> [--camera N]
          julia capture_iriun.jl --test
          julia capture_iriun.jl --list
        
        COMANDOS:
          --single    Capturar uma única imagem
          --multiple  Capturar múltiplas imagens
          --test      Testar todas as câmeras
          --list      Listar câmeras disponíveis
        
        OPÇÕES:
          --camera N  Usar câmera específica (índice N)
        
        EXEMPLOS:
          # Listar câmeras
          julia capture_iriun.jl --list
          
          # Testar câmeras
          julia capture_iriun.jl --test
          
          # Captura única (detecta Iriun automaticamente)
          julia capture_iriun.jl --single foto.jpg
          
          # Captura única com câmera específica
          julia capture_iriun.jl --single foto.jpg --camera 2
          
          # Captura múltipla
          julia capture_iriun.jl --multiple "João Silva" "../dados/fotos_train" 15
          
          # Captura múltipla com câmera específica
          julia capture_iriun.jl --multiple "Maria" "../dados/fotos_train" 15 --camera 2
        
        DICAS IRIUN WEBCAM:
          1. Inicie o app Iriun no celular
          2. Conecte via USB ou WiFi (mesma rede)
          3. No Linux, verifique: ls -l /dev/video*
          4. Iriun geralmente aparece como /dev/video2 ou superior
          5. Se não funcionar: sudo systemctl restart iriunwebcam
        """)
        return
    end
    
    # Processar comandos
    if ARGS[1] == "--list"
        list_all_cameras()
        
    elseif ARGS[1] == "--test"
        test_cameras()
        
    elseif ARGS[1] == "--single"
        if length(ARGS) < 2
            println("❌ Uso: julia capture_iriun.jl --single <output.jpg> [--camera N]")
            return
        end
        
        output_path = ARGS[2]
        camera_idx = nothing
        
        # Verificar se tem --camera
        if length(ARGS) >= 4 && ARGS[3] == "--camera"
            camera_idx = parse(Int, ARGS[4])
        end
        
        capture_single_image_iriun(output_path; camera_index=camera_idx)
        
    elseif ARGS[1] == "--multiple"
        if length(ARGS) < 4
            println("❌ Uso: julia capture_iriun.jl --multiple <nome> <diretorio> <num> [--camera N]")
            return
        end
        
        person_name = ARGS[2]
        output_dir = ARGS[3]
        num_images = parse(Int, ARGS[4])
        camera_idx = nothing
        
        # Verificar se tem --camera
        if length(ARGS) >= 6 && ARGS[5] == "--camera"
            camera_idx = parse(Int, ARGS[6])
        end
        
        capture_multiple_images_iriun(person_name, output_dir, num_images; camera_index=camera_idx)
        
    else
        println("❌ Comando desconhecido: $(ARGS[1])")
        println("Use sem argumentos para ver a ajuda")
    end
end

# ============================================================================
# EXECUÇÃO
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

                if frame !== nothing
                    println("  ✅ Câmera funcional encontrada no índice: $camera_idx")
                    
                    # Verificar se é Iriun (tamanho típico ou nome)
                    # Iriun geralmente usa resoluções específicas
                    height, width = size(frame)[1:2]
                    
                    if width >= 640 && height >= 480
                        println("  📱 Possível Iriun Webcam (resolução: $(width)x$(height))")
                        return camera_idx
                    end
                end
            end
        catch e
            continue
        end
    end
    
    println("  ⚠️  Iriun Webcam não detectada automaticamente")
    return nothing
end

"""
    list_all_cameras() -> Vector{Int}

Lista todas as câmeras disponíveis no sistema.
"""
function list_all_cameras()
    println("\n🎥 Listando todas as câmeras disponíveis...")
    available = Int[]
    
    for i in 0:10
        try
            cam = VideoIO.opencamera(i)
            if cam !== nothing
                frame = read(cam)
                close(cam)