# projeto: cnncheckin
# file: cnncheckin/src/cnncheckin_webcam.jl
# descrição: Módulo para captura de imagens via webcam

module CNNCheckinWebcam

using Images
using FileIO
using Dates
using ImageView
using VideoIO

export capture_single_image,
       capture_multiple_images,
       capture_training_session,
       list_available_cameras,
       preview_camera

# ============================================================================
# CONFIGURAÇÕES DA WEBCAM
# ============================================================================

const DEFAULT_CAMERA_INDEX = 0
const CAPTURE_WIDTH = 640
const CAPTURE_HEIGHT = 480
const PREVIEW_WINDOW_NAME = "CNNCheckin - Preview"

# ============================================================================
# FUNÇÕES DE CÂMERA
# ============================================================================

"""
    list_available_cameras() -> Vector{Int}

Lista os índices das câmeras disponíveis no sistema.
"""
function list_available_cameras()
    println("\n🎥 Detectando câmeras disponíveis...")
    available = Int[]
    
    for i in 0:5  # Testa até 5 câmeras
        try
            cam = VideoIO.opencamera(i)
            if cam !== nothing
                push!(available, i)
                println("  ✓ Câmera $i detectada")
                close(cam)
            end
        catch e
            continue
        end
    end
    
    if isempty(available)
        println("  ⚠️  Nenhuma câmera detectada")
    else
        println("\n✅ Total de câmeras encontradas: $(length(available))")
    end
    
    return available
end

"""
    preview_camera(camera_index::Int=DEFAULT_CAMERA_INDEX; duration::Int=5)

Abre preview da câmera por alguns segundos.
"""
function preview_camera(camera_index::Int=DEFAULT_CAMERA_INDEX; duration::Int=5)
    println("\n📹 Abrindo preview da câmera $camera_index...")
    println("   Preview durará $duration segundos")
    
    try
        cam = VideoIO.opencamera(camera_index)
        
        start_time = time()
        frame_count = 0
        
        while (time() - start_time) < duration
            frame = read(cam)
            if frame !== nothing
                # Converter para RGB se necessário
                img = RGB.(frame)
                
                # Mostrar frame
                if frame_count == 0
                    imshow(img)
                end
                
                frame_count += 1
            end
            sleep(0.033)  # ~30 FPS
        end
        
        close(cam)
        println("✅ Preview finalizado - $frame_count frames capturados")
        
    catch e
        println("❌ Erro ao abrir câmera: $e")
        return false
    end
    
    return true
end

# ============================================================================
# CAPTURA DE IMAGENS
# ============================================================================

"""
    capture_single_image(output_path::String; camera_index::Int=DEFAULT_CAMERA_INDEX,
                        show_preview::Bool=true, countdown::Int=3) -> Bool

Captura uma única imagem da webcam.

# Argumentos
- `output_path`: Caminho onde a imagem será salva
- `camera_index`: Índice da câmera (padrão: 0)
- `show_preview`: Se deve mostrar preview antes de capturar
- `countdown`: Tempo de contagem regressiva em segundos

# Retorna
`true` se a captura foi bem-sucedida, `false` caso contrário
"""
function capture_single_image(output_path::String; 
                             camera_index::Int=DEFAULT_CAMERA_INDEX,
                             show_preview::Bool=true,
                             countdown::Int=3)
    println("\n📸 Iniciando captura de imagem...")
    println("   Câmera: $camera_index")
    println("   Destino: $output_path")
    
    try
        # Abrir câmera
        cam = VideoIO.opencamera(camera_index, 
                                width=CAPTURE_WIDTH, 
                                height=CAPTURE_HEIGHT)
        
        if cam === nothing
            println("❌ Não foi possível abrir a câmera $camera_index")
            return false
        end
        
        println("✅ Câmera aberta com sucesso")
        
        # Preview e countdown
        if show_preview
            println("\n⏱️  Preparando captura em $countdown segundos...")
            
            for i in countdown:-1:1
                println("   $i...")
                frame = read(cam)
                if frame !== nothing
                    img = RGB.(frame)
                    imshow(img)
                end
                sleep(1)
            end
        end
        
        # Capturar frame
        println("📸 Capturando...")
        frame = read(cam)
        
        if frame === nothing
            println("❌ Falha ao capturar frame")
            close(cam)
            return false
        end
        
        # Converter e salvar
        img = RGB.(frame)
        
        # Criar diretório se não existir
        output_dir = dirname(output_path)
        if !isempty(output_dir) && !isdir(output_dir)
            mkpath(output_dir)
        end
        
        save(output_path, img)
        println("✅ Imagem salva: $output_path")
        
        # Fechar câmera
        close(cam)
        
        return true
        
    catch e
        println("❌ Erro durante captura: $e")
        return false
    end
end

"""
    capture_multiple_images(person_name::String, output_dir::String, 
                           num_images::Int=10; camera_index::Int=DEFAULT_CAMERA_INDEX,
                           delay_between::Int=2) -> Int

Captura múltiplas imagens de uma pessoa.

# Retorna
Número de imagens capturadas com sucesso
"""
function capture_multiple_images(person_name::String, 
                                output_dir::String, 
                                num_images::Int=10;
                                camera_index::Int=DEFAULT_CAMERA_INDEX,
                                delay_between::Int=2)
    println("\n📸 Captura múltipla de imagens")
    println("="^60)
    println("   Pessoa: $person_name")
    println("   Quantidade: $num_images imagens")
    println("   Intervalo: $delay_between segundos")
    println("   Destino: $output_dir")
    println("="^60)
    
    # Criar diretório se não existir
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    
    # Abrir câmera
    cam = nothing
    try
        cam = VideoIO.opencamera(camera_index,
                                width=CAPTURE_WIDTH,
                                height=CAPTURE_HEIGHT)
        
        if cam === nothing
            println("❌ Não foi possível abrir a câmera")
            return 0
        end
        
        println("✅ Câmera inicializada")
        
    catch e
        println("❌ Erro ao abrir câmera: $e")
        return 0
    end
    
    captured = 0
    
    println("\n🎬 Iniciando sequência de capturas...")
    println("💡 Dica: Varie a posição e expressão entre as capturas\n")
    
    for i in 1:num_images
        try
            println("[$i/$num_images] Preparando captura...")
            
            # Countdown curto
            for j in delay_between:-1:1
                print("   $j... ")
                flush(stdout)
                
                # Ler frame para preview
                frame = read(cam)
                if frame !== nothing
                    img = RGB.(frame)
                    if i == 1 || j == delay_between
                        imshow(img)
                    end
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
            
            # Salvar com timestamp
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
    
    println("\n" * "="^60)
    println("✅ Sessão finalizada!")
    println("   Imagens capturadas: $captured/$num_images")
    println("   Taxa de sucesso: $(round(captured/num_images*100, digits=1))%")
    println("="^60)
    
    return captured
end

"""
    capture_training_session(person_name::String, output_dir::String;
                            num_images::Int=15, camera_index::Int=DEFAULT_CAMERA_INDEX) -> Bool

Captura imagens para treinamento com instruções interativas.
"""
function capture_training_session(person_name::String, 
                                 output_dir::String;
                                 num_images::Int=15,
                                 camera_index::Int=DEFAULT_CAMERA_INDEX)
    println("\n" * "="^70)
    println("🎓 SESSÃO DE CAPTURA PARA TREINAMENTO")
    println("="^70)
    println("\n👤 Pessoa: $person_name")
    println("📁 Diretório: $output_dir")
    println("📸 Número de fotos: $num_images")
    
    println("\n💡 INSTRUÇÕES IMPORTANTES:")
    println("   1. Posicione-se em frente à câmera com boa iluminação")
    println("   2. Mantenha o rosto centralizado e visível")
    println("   3. Varie a expressão facial entre as capturas")
    println("   4. Varie levemente o ângulo da cabeça")
    println("   5. Evite óculos escuros ou objetos que cubram o rosto")
    
    println("\n⏸️  Pressione ENTER para iniciar ou 'q' para cancelar...")
    response = readline()
    
    if lowercase(strip(response)) == "q"
        println("❌ Sessão cancelada")
        return false
    end
    
    # Dividir capturas em grupos
    poses = [
        ("frontal", 5),
        ("virado levemente à esquerda", 3),
        ("virado levemente à direita", 3),
        ("com expressões variadas", 4)
    ]
    
    total_captured = 0
    
    for (pose_desc, pose_count) in poses
        println("\n" * "─"^70)
        println("📸 Próxima pose: $pose_desc ($pose_count fotos)")
        println("─"^70)
        println("⏸️  Pressione ENTER quando estiver pronto...")
        readline()
        
        captured = capture_multiple_images(
            person_name,
            output_dir,
            pose_count;
            camera_index=camera_index,
            delay_between=2
        )
        
        total_captured += captured
    end
    
    println("\n" * "="^70)
    println("🎉 SESSÃO DE TREINAMENTO CONCLUÍDA!")
    println("="^70)
    println("   Total de imagens capturadas: $total_captured/$num_images")
    
    if total_captured >= div(num_images * 3, 4)
        println("   ✅ Quantidade suficiente para treinamento!")
        return true
    else
        println("   ⚠️  Poucas imagens capturadas. Recomenda-se repetir.")
        return false
    end
end

# ============================================================================
# FUNÇÕES DE UTILIDADE
# ============================================================================

"""
    check_camera_available(camera_index::Int=DEFAULT_CAMERA_INDEX) -> Bool

Verifica se a câmera está disponível.
"""
function check_camera_available(camera_index::Int=DEFAULT_CAMERA_INDEX)
    try
        cam = VideoIO.opencamera(camera_index)
        if cam !== nothing
            close(cam)
            return true
        end
    catch
    end
    return false
end

"""
    get_recommended_camera() -> Int

Retorna o índice da câmera recomendada (primeira disponível).
"""
function get_recommended_camera()
    cameras = list_available_cameras()
    return isempty(cameras) ? DEFAULT_CAMERA_INDEX : cameras[1]
end

end  # module CNNCheckinWebcam