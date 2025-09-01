# projeto: cnncheckin
# file: cnncheckin/src/cnncheckin_acount_core.jl
# Parte 1: Funções principais e configuração do sistema
# Funções principais do sistema de captura de fotos

using VideoIO
using Images
using FileIO
using Dates

# Sistema de display com múltiplos fallbacks
graphics_available = Dict()

# Try to load optional graphics packages
try
    using Gtk, GLib, Cairo
    graphics_available["gtk"] = true
catch
    graphics_available["gtk"] = false
    println("⚠️ GTK não disponível - interface gráfica limitada")
end

try
    using Plots
    graphics_available["plots"] = true
catch
    graphics_available["plots"] = false
end

DISPLAY_AVAILABLE = graphics_available["gtk"]
DISPLAY_TYPE = DISPLAY_AVAILABLE ? "gtk" : "console"

"""
    criar_diretorio(caminho::String)

Cria um diretório se ele não existir.
"""
function criar_diretorio(caminho::String)
    if !isdir(caminho)
        mkpath(caminho)
        println("📁 Diretório criado: $caminho")
        return true
    end
    return false
end

"""
    listar_fotos(pasta::String) -> Vector{String}

Lista todas as fotos válidas em uma pasta.
"""
function listar_fotos(pasta::String)
    if !isdir(pasta)
        return String[]
    end
    
    extensoes_validas = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    fotos = String[]
    
    for arquivo in readdir(pasta)
        caminho_completo = joinpath(pasta, arquivo)
        if isfile(caminho_completo)
            _, ext = splitext(lowercase(arquivo))
            if ext in extensoes_validas
                push!(fotos, caminho_completo)
            end
        end
    end
    
    return sort(fotos)
end

"""
    info_foto(caminho_foto::String) -> String

Retorna informações detalhadas sobre uma foto.
"""
function info_foto(caminho_foto::String)
    if !isfile(caminho_foto)
        return "❌ Arquivo não encontrado"
    end
    
    try
        img = load(caminho_foto)
        nome_arquivo = basename(caminho_foto)
        tamanho = size(img)
        
        # Obter tamanho do arquivo
        tamanho_arquivo = filesize(caminho_foto)
        tamanho_kb = round(tamanho_arquivo / 1024, digits=2)
        tamanho_mb = round(tamanho_arquivo / (1024*1024), digits=2)
        
        # Informação sobre tipo de imagem
        tipo_img = typeof(img)
        
        # Extrair timestamp do nome se possível
        timestamp_info = ""
        if occursin("_", nome_arquivo)
            partes = split(nome_arquivo, "_")
            if length(partes) >= 3
                data_parte = partes[2]
                hora_parte = split(partes[3], ".")[1]
                timestamp_info = "\n📅 Data/Hora: $(replace(data_parte, "-" => "/")) $(replace(hora_parte, "-" => ":"))"
            end
        end
        
        tamanho_str = tamanho_mb > 1 ? "$(tamanho_mb) MB" : "$(tamanho_kb) KB"
        
        return """
📷 Arquivo: $nome_arquivo
📐 Dimensões: $(tamanho[2])x$(tamanho[1]) pixels
💾 Tamanho: $tamanho_str
🎨 Tipo: $tipo_img$timestamp_info
📂 Caminho: $caminho_foto
        """
    catch e
        return "❌ Erro ao ler informações: $e"
    end
end

"""
    verificar_webcam() -> (Bool, Int)

Verifica se há webcam disponível e retorna (sucesso, índice).
Versão corrigida para tratar tipos de erro adequadamente.
"""
function verificar_webcam()
    println("🔍 Verificando webcam...")
    
    try
        # Tentar diferentes índices de câmera
        for i in 0:2
            try
                camera = VideoIO.opencamera(i)
                println("✅ Webcam encontrada no índice $i")
                
                # Testar captura
                frame = read(camera)
                if frame !== nothing
                    println("✅ Captura de frame funcionando")
                    println("📐 Resolução: $(size(frame))")
                end
                
                close(camera)
                return true, i
            catch e
                if i == 0
                    # Corrigido: usar string(typeof(e)) em vez de typeof(e).__name__
                    println("⚠️ Webcam padrão (índice 0): $(string(typeof(e)))")
                end
                continue
            end
        end
        
        println("❌ Nenhuma webcam encontrada")
        return false, -1
        
    catch e
        println("❌ Erro geral na verificação: $e")
        return false, -1
    end
end

"""
    capturar_frame(camera) -> Union{Nothing, Matrix}

Captura um frame da webcam com múltiplas tentativas.
"""
function capturar_frame(camera)
    melhor_frame = nothing
    for tentativa in 1:3
        try
            frame = read(camera)
            if frame !== nothing
                melhor_frame = frame
                break
            end
        catch e
            println("⚠️ Tentativa $tentativa falhou: $e")
        end
        sleep(0.1)
    end
    return melhor_frame
end

"""
    salvar_foto(frame, pasta::String, prefixo::String="foto") -> String

Salva um frame como foto e retorna o caminho do arquivo.
"""
function salvar_foto(frame, pasta::String, prefixo::String="foto")
    criar_diretorio(pasta)
    
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS-sss")
    nome_arquivo = "$(prefixo)_$timestamp.jpg"
    caminho_completo = joinpath(pasta, nome_arquivo)
    
    try
        save(caminho_completo, frame)
        tamanho_kb = round(filesize(caminho_completo) / 1024, digits=1)
        println("✅ Foto salva: $nome_arquivo ($(tamanho_kb) KB)")
        return caminho_completo
    catch e
        println("❌ Erro ao salvar foto: $e")
        return ""
    end
end

"""
    mostrar_estatisticas(pastas::Vector{String})

Mostra estatísticas das pastas de fotos.
"""
function mostrar_estatisticas(pastas::Vector{String})
    println("\n📊 === ESTATÍSTICAS ===")
    
    total_fotos = 0
    total_tamanho = 0.0
    
    for pasta in pastas
        if isdir(pasta)
            fotos = listar_fotos(pasta)
            tamanho_pasta = sum(filesize(foto) for foto in fotos) / (1024*1024)
            
            println("\n📁 $pasta:")
            println("   📸 Fotos: $(length(fotos))")
            println("   💾 Tamanho: $(round(tamanho_pasta, digits=2)) MB")
            
            if !isempty(fotos)
                primeira = basename(first(sort(fotos)))
                ultima = basename(last(sort(fotos)))
                println("   📅 Primeira: $primeira")
                println("   📅 Última: $ultima")
            end
            
            total_fotos += length(fotos)
            total_tamanho += tamanho_pasta
        else
            println("📁 $pasta: (não existe)")
        end
    end
    
    println("\n🏆 TOTAL GERAL:")
    println("   📸 Fotos: $total_fotos")
    println("   💾 Tamanho: $(round(total_tamanho, digits=2)) MB")
end

"""
    abrir_com_visualizador_externo(caminho_foto::String) -> Bool

Abre foto com visualizador externo do sistema.
"""
function abrir_com_visualizador_externo(caminho_foto::String)
    if !isfile(caminho_foto)
        println("❌ Arquivo não encontrado")
        return false
    end
    
    visualizadores = []
    
    if Sys.islinux()
        linux_viewers = [
            ("eog", "Eye of GNOME"),
            ("gwenview", "KDE Gwenview"), 
            ("feh", "Feh (leve)"),
            ("display", "ImageMagick"),
            ("xdg-open", "Padrão do sistema")
        ]
        visualizadores = linux_viewers
    elseif Sys.iswindows()
        visualizadores = [("start", "Windows padrão")]
    elseif Sys.isapple()
        visualizadores = [("open", "macOS padrão")]
    end
    
    for (comando, nome) in visualizadores
        try
            if Sys.islinux()
                run(pipeline(`which $comando`, devnull), wait=true)
                
                if comando == "xdg-open"
                    run(`$comando $caminho_foto`, wait=false)
                else
                    run(`$comando $caminho_foto`, wait=false)
                end
            elseif Sys.iswindows()
                run(`cmd /c start "" "$caminho_foto"`, wait=false)
            elseif Sys.isapple()
                run(`open $caminho_foto`, wait=false)
            end
            
            println("🖼️ Imagem aberta com $nome")
            return true
            
        catch
            continue
        end
    end
    
    println("❌ Nenhum visualizador disponível encontrado")
    println("💡 Instale um visualizador:")
    println("   sudo apt install eog gwenview feh imagemagick")
    return false
end

"""
    criar_backup(pastas::Vector{String}) -> String

Cria backup das fotos e retorna o caminho do backup.
"""
function criar_backup(pastas::Vector{String})
    println("\n📦 === CRIAR BACKUP ===")
    
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    pasta_backup = "backup_fotos_$timestamp"
    
    criar_diretorio(pasta_backup)
    
    total_copiadas = 0
    
    for pasta in pastas
        if isdir(pasta)
            fotos = listar_fotos(pasta)
            if !isempty(fotos)
                pasta_destino = joinpath(pasta_backup, pasta)
                criar_diretorio(pasta_destino)
                
                println("📂 Copiando $pasta ($(length(fotos)) fotos)...")
                
                for foto in fotos
                    try
                        destino = joinpath(pasta_destino, basename(foto))
                        cp(foto, destino)
                        total_copiadas += 1
                    catch e
                        println("⚠️ Erro ao copiar $(basename(foto)): $e")
                    end
                end
            end
        end
    end
    
    println("✅ Backup criado: $pasta_backup")
    println("📸 Total de fotos copiadas: $total_copiadas")
    return pasta_backup
end

"""
    limpar_pasta(pasta::String) -> Int

Remove todas as fotos de uma pasta e retorna quantas foram removidas.
"""
function limpar_pasta(pasta::String)
    if !isdir(pasta)
        return 0
    end
    
    fotos = listar_fotos(pasta)
    removidas = 0
    
    for foto in fotos
        try
            rm(foto)
            removidas += 1
        catch e
            println("⚠️ Erro ao remover $(basename(foto)): $e")
        end
    end
    
    return removidas
end

# Configurações padrão do sistema
const CONFIG_PADRAO = Dict(
    "pasta_fotos" => "fotos_rosto",
    "num_fotos_padrao" => 5,
    "intervalo_padrao" => 3,
    "qualidade_jpg" => 95,
    "prefixo_arquivo" => "foto"
)

"""
    get_config(chave::String)

Obtém configuração padrão.
"""
function get_config(chave::String)
    return get(CONFIG_PADRAO, chave, nothing)
end

# Variáveis globais para estado da aplicação
mutable struct AppState
    camera
    camera_index::Int
    fotos_capturadas::Vector{String}
    pasta_atual::String
    webcam_ativa::Bool
    
    AppState() = new(nothing, -1, String[], get_config("pasta_fotos"), false)
end

# Instância global do estado
const app_state = AppState()

"""
    inicializar_webcam() -> Bool

Inicializa a webcam e retorna sucesso.
"""
function inicializar_webcam()
    if app_state.webcam_ativa
        return true
    end
    
    webcam_ok, camera_index = verificar_webcam()
    if !webcam_ok
        return false
    end
    
    try
        app_state.camera = VideoIO.opencamera(camera_index)
        app_state.camera_index = camera_index
        app_state.webcam_ativa = true
        
        # Warm-up da câmera
        for _ in 1:3
            try
                read(app_state.camera)
                sleep(0.2)
            catch
                break
            end
        end
        
        println("📹 Webcam inicializada (índice: $camera_index)")
        return true
    catch e
        println("❌ Erro ao inicializar webcam: $e")
        return false
    end
end

"""
    fechar_webcam()

Fecha a webcam se estiver ativa.
"""
function fechar_webcam()
    if app_state.webcam_ativa && app_state.camera !== nothing
        try
            close(app_state.camera)
            app_state.webcam_ativa = false
            app_state.camera = nothing
            println("📹 Webcam fechada")
        catch e
            println("⚠️ Erro ao fechar webcam: $e")
        end
    end
end

"""
    capturar_foto_atual() -> String

Captura uma foto da webcam ativa e retorna o caminho do arquivo.
"""
function capturar_foto_atual()
    if !app_state.webcam_ativa || app_state.camera === nothing
        println("❌ Webcam não está ativa")
        return ""
    end
    
    frame = capturar_frame(app_state.camera)
    if frame === nothing
        println("❌ Falha ao capturar frame")
        return ""
    end
    
    caminho = salvar_foto(frame, app_state.pasta_atual)
    if !isempty(caminho)
        push!(app_state.fotos_capturadas, caminho)
    end
    
    return caminho
end