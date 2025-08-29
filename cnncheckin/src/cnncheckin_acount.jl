# projeto: cnncheckin
# file: cnncheckin/src/cnncheckin_acount.jl
# Versão corrigida com verificação de dependências e widgets GTK modernos
 
# Versão corrigida com compatibilidade GTK

using Pkg

# Lista de pacotes necessários
required_packages = ["Gtk", "Cairo", "VideoIO", "Images", "FileIO", "Dates"]

println("🔧 Verificando dependências...")
for pkg in required_packages
    try
        eval(Meta.parse("using $pkg"))
        println("✅ $pkg - OK")
    catch
        println("📦 Instalando $pkg...")
        try
            Pkg.add(pkg)
            eval(Meta.parse("using $pkg"))
            println("✅ $pkg instalado e carregado")
        catch e
            println("❌ Erro ao instalar $pkg: $e")
        end
    end
end

using Gtk, Cairo, VideoIO, Images, FileIO, Dates
import GLib

# Configurações padrão
const CONFIG = Dict(
    "pasta_fotos" => "fotos_rosto",
    "num_fotos_padrao" => 5,
    "intervalo_padrao" => 3
)

# Estado da aplicação
mutable struct AppState
    camera
    camera_index::Int
    webcam_ativa::Bool
    pasta_atual::String
    fotos_capturadas::Vector{String}
    
    AppState() = new(nothing, -1, false, CONFIG["pasta_fotos"], String[])
end

const app_state = AppState()

# GUI Components
mutable struct CNNCheckInGUI
    window::GtkWindow
    preview_area::GtkDrawingArea  # Voltando para GtkDrawingArea
    btn_iniciar::GtkButton
    btn_parar::GtkButton
    btn_capturar::GtkButton
    btn_visualizar::GtkButton
    entry_pasta::GtkEntry
    label_status::GtkLabel
    label_contador::GtkLabel
    timer_id::Union{Int, Nothing}
    
    CNNCheckInGUI() = new()
end

const gui = CNNCheckInGUI()

"""
    criar_diretorio(caminho::String)
Cria diretório se não existir.
"""
function criar_diretorio(caminho::String)
    if !isdir(caminho)
        mkpath(caminho)
        println("📁 Diretório criado: $caminho")
    end
end

"""
    verificar_webcam() -> (Bool, Int)
Verifica webcam disponível.
"""
function verificar_webcam()
    println("🔍 Verificando webcam...")
    
    for i in 0:2
        try
            camera = VideoIO.opencamera(i)
            frame = read(camera)
            if frame !== nothing
                println("✅ Webcam encontrada no índice $i")
                close(camera)
                return true, i
            end
            close(camera)
        catch e
            continue
        end
    end
    
    println("❌ Nenhuma webcam encontrada")
    return false, -1
end

"""
    inicializar_webcam() -> Bool
Inicializa webcam.
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
                sleep(0.1)
            catch
                break
            end
        end
        
        println("📹 Webcam inicializada")
        return true
    catch e
        println("❌ Erro ao inicializar webcam: $e")
        return false
    end
end

"""
    fechar_webcam()
Fecha webcam.
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
    capturar_frame()
Captura frame da webcam.
"""
function capturar_frame()
    if !app_state.webcam_ativa || app_state.camera === nothing
        return nothing
    end
    
    try
        return read(app_state.camera)
    catch
        return nothing
    end
end

"""
    salvar_foto(frame, pasta::String) -> String
Salva foto e retorna caminho.
"""
function salvar_foto(frame, pasta::String)
    criar_diretorio(pasta)
    
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS-sss")
    nome_arquivo = "foto_$timestamp.jpg"
    caminho_completo = joinpath(pasta, nome_arquivo)
    
    try
        save(caminho_completo, frame)
        println("✅ Foto salva: $nome_arquivo")
        return caminho_completo
    catch e
        println("❌ Erro ao salvar: $e")
        return ""
    end
end

"""
    listar_fotos(pasta::String) -> Vector{String}
Lista fotos válidas na pasta.
"""
function listar_fotos(pasta::String)
    if !isdir(pasta)
        return String[]
    end
    
    extensoes = [".jpg", ".jpeg", ".png", ".bmp"]
    fotos = String[]
    
    for arquivo in readdir(pasta)
        caminho = joinpath(pasta, arquivo)
        if isfile(caminho)
            _, ext = splitext(lowercase(arquivo))
            if ext in extensoes
                push!(fotos, caminho)
            end
        end
    end
    
    return sort(fotos)
end

"""
    abrir_visualizador(foto::String)
Abre foto no visualizador do sistema.
"""
function abrir_visualizador(foto::String)
    if !isfile(foto)
        return false
    end
    
    try
        if Sys.islinux()
            run(`xdg-open $foto`, wait=false)
        elseif Sys.iswindows()
            run(`cmd /c start "" "$foto"`, wait=false)
        elseif Sys.isapple()
            run(`open $foto`, wait=false)
        end
        println("🖼️ Foto aberta")
        return true
    catch
        println("❌ Erro ao abrir visualizador")
        return false
    end
end

# === CALLBACKS DA GUI ===

"""
    on_draw_preview(widget, cr)
Callback para desenhar preview.
"""
function on_draw_preview(widget, cr)
    # Fundo preto
    Cairo.set_source_rgb(cr, 0, 0, 0)
    Cairo.paint(cr)
    
    if !app_state.webcam_ativa
        # Texto quando webcam não ativa
        Cairo.set_source_rgb(cr, 1, 1, 1)
        Cairo.select_font_face(cr, "Arial", Cairo.FONT_SLANT_NORMAL, Cairo.FONT_WEIGHT_BOLD)
        Cairo.set_font_size(cr, 16)
        
        text = "Webcam não iniciada"
        text_ext = Cairo.text_extents(cr, text)
        
        w = Gtk.allocated_width(widget)
        h = Gtk.allocated_height(widget)
        x = (w - text_ext.width) / 2
        y = (h + text_ext.height) / 2
        
        Cairo.move_to(cr, x, y)
        Cairo.show_text(cr, text)
    else
        # Desenhar frame se disponível
        frame = capturar_frame()
        if frame !== nothing
            try
                desenhar_frame_cairo(cr, frame, widget)
            catch e
                # Em caso de erro, mostrar mensagem
                Cairo.set_source_rgb(cr, 1, 1, 0)
                Cairo.move_to(cr, 10, 30)
                Cairo.show_text(cr, "Erro no preview")
            end
        end
    end
    
    return false
end

"""
    desenhar_frame_cairo(cr, frame, widget)
Desenha frame no contexto Cairo.
"""
function desenhar_frame_cairo(cr, frame, widget)
    try
        # Converter para RGB se necessário
        if eltype(frame) != RGB{N0f8}
            frame = RGB.(frame)
        end
        
        h, w = size(frame)
        widget_w = Gtk.allocated_width(widget)
        widget_h = Gtk.allocated_height(widget)
        
        # Calcular escala mantendo aspecto
        scale = min(widget_w / w, widget_h / h)
        new_w = round(Int, w * scale)
        new_h = round(Int, h * scale)
        
        # Redimensionar se necessário
        if new_w != w || new_h != h
            frame = imresize(frame, (new_h, new_w))
        end
        
        # Preparar dados da imagem para Cairo
        img_data = zeros(UInt8, new_h * new_w * 4)
        
        idx = 1
        for i in 1:new_h
            for j in 1:new_w
                pixel = frame[i, j]
                img_data[idx] = round(UInt8, blue(pixel) * 255)   # B
                img_data[idx+1] = round(UInt8, green(pixel) * 255) # G
                img_data[idx+2] = round(UInt8, red(pixel) * 255)   # R
                img_data[idx+3] = 255  # A
                idx += 4
            end
        end
        
        # Criar surface Cairo
        stride = Cairo.format_stride_for_width(Cairo.FORMAT_RGB24, new_w)
        surface = Cairo.CairoImageSurface(img_data, Cairo.FORMAT_RGB24, new_w, new_h, stride)
        
        # Centralizar imagem
        offset_x = (widget_w - new_w) / 2
        offset_y = (widget_h - new_h) / 2
        
        Cairo.save(cr)
        Cairo.translate(cr, offset_x, offset_y)
        Cairo.set_source_surface(cr, surface, 0, 0)
        Cairo.paint(cr)
        Cairo.restore(cr)
        
    catch e
        println("⚠️ Erro ao desenhar frame: $e")
    end
end

"""
    atualizar_preview()
Timer callback para atualizar preview.
"""
function atualizar_preview()
    if app_state.webcam_ativa && gui.preview_area !== nothing
        Gtk.queue_draw(gui.preview_area)
    end
    return app_state.webcam_ativa
end

"""
    on_iniciar_clicked(button)
Callback botão iniciar.
"""
function on_iniciar_clicked(button)
    if inicializar_webcam()
        # Iniciar timer para preview
        gui.timer_id = GLib.g_timeout_add(100, atualizar_preview)
        
        # Atualizar interface
        Gtk.set_gtk_property!(gui.btn_iniciar, :sensitive, false)
        Gtk.set_gtk_property!(gui.btn_parar, :sensitive, true)
        Gtk.set_gtk_property!(gui.btn_capturar, :sensitive, true)
        Gtk.set_gtk_property!(gui.label_status, :label, "✅ Webcam ativa")
    else
        Gtk.set_gtk_property!(gui.label_status, :label, "❌ Erro ao iniciar webcam")
    end
end

"""
    on_parar_clicked(button)
Callback botão parar.
"""
function on_parar_clicked(button)
    if gui.timer_id !== nothing
        GLib.g_source_remove(gui.timer_id)
        gui.timer_id = nothing
    end
    
    fechar_webcam()
    
    # Atualizar interface
    Gtk.set_gtk_property!(gui.btn_iniciar, :sensitive, true)
    Gtk.set_gtk_property!(gui.btn_parar, :sensitive, false)
    Gtk.set_gtk_property!(gui.btn_capturar, :sensitive, false)
    Gtk.set_gtk_property!(gui.label_status, :label, "🔴 Webcam desligada")
    
    # Redesenhar preview
    Gtk.queue_draw(gui.preview_area)
end

"""
    on_capturar_clicked(button)
Callback botão capturar.
"""
function on_capturar_clicked(button)
    frame = capturar_frame()
    if frame !== nothing
        pasta = Gtk.get_gtk_property(gui.entry_pasta, :text, String)
        app_state.pasta_atual = pasta
        
        caminho = salvar_foto(frame, pasta)
        if !isempty(caminho)
            push!(app_state.fotos_capturadas, caminho)
            contador = length(app_state.fotos_capturadas)
            Gtk.set_gtk_property!(gui.label_contador, :label, "Fotos: $contador")
            Gtk.set_gtk_property!(gui.label_status, :label, "📸 Foto: $(basename(caminho))")
        end
    else
        Gtk.set_gtk_property!(gui.label_status, :label, "❌ Erro ao capturar foto")
    end
end

"""
    on_visualizar_clicked(button)
Callback botão visualizar.
"""
function on_visualizar_clicked(button)
    pasta = Gtk.get_gtk_property(gui.entry_pasta, :text, String)
    fotos = listar_fotos(pasta)
    
    if !isempty(fotos)
        abrir_visualizador(last(fotos))  # Abre a última foto
    else
        Gtk.set_gtk_property!(gui.label_status, :label, "❌ Nenhuma foto encontrada")
    end
end

"""
    on_destroy(widget)
Callback destruição da janela.
"""
function on_destroy(widget)
    if gui.timer_id !== nothing
        GLib.g_source_remove(gui.timer_id)
    end
    fechar_webcam()
    Gtk.gtk_quit()
end

"""
    criar_interface()
Cria interface principal.
"""
function criar_interface()
    # Janela principal
    gui.window = GtkWindow("CNN Check-In", 750, 550)
    Gtk.set_gtk_property!(gui.window, :resizable, false)
    
    # Layout principal
    vbox = GtkBox(:v, 10)
    Gtk.set_gtk_property!(vbox, :margin_left, 10)
    Gtk.set_gtk_property!(vbox, :margin_right, 10)
    Gtk.set_gtk_property!(vbox, :margin_top, 10)
    Gtk.set_gtk_property!(vbox, :margin_bottom, 10)
    
    # Título
    title = GtkLabel("CNN Check-In - Captura Facial")
    # Remover markup - usar texto simples
    push!(vbox, title)
    
    # Container horizontal
    hbox = GtkBox(:h, 10)
    
    # === PREVIEW ===
    frame_preview = GtkFrame("Preview da Webcam")
    Gtk.set_gtk_property!(frame_preview, :width_request, 400)
    Gtk.set_gtk_property!(frame_preview, :height_request, 300)
    
    gui.preview_area = GtkDrawingArea()
    Gtk.set_gtk_property!(gui.preview_area, :width_request, 380)
    Gtk.set_gtk_property!(gui.preview_area, :height_request, 280)
    
    # Conectar callback de desenho
    Gtk.signal_connect(on_draw_preview, gui.preview_area, "draw")
    
    push!(frame_preview, gui.preview_area)
    push!(hbox, frame_preview)
    
    # === CONTROLES ===
    vbox_ctrl = GtkBox(:v, 10)
    Gtk.set_gtk_property!(vbox_ctrl, :width_request, 320)
    
    # Configurações
    frame_config = GtkFrame("Configurações")
    vbox_config = GtkBox(:v, 5)
    Gtk.set_gtk_property!(vbox_config, :margin_left, 5)
    Gtk.set_gtk_property!(vbox_config, :margin_right, 5)
    Gtk.set_gtk_property!(vbox_config, :margin_top, 5)
    Gtk.set_gtk_property!(vbox_config, :margin_bottom, 5)
    
    # Pasta de destino
    hbox_pasta = GtkBox(:h, 5)
    label_pasta = GtkLabel("Pasta:")
    Gtk.set_gtk_property!(label_pasta, :width_request, 50)
    push!(hbox_pasta, label_pasta)
    
    gui.entry_pasta = GtkEntry()
    Gtk.set_gtk_property!(gui.entry_pasta, :text, CONFIG["pasta_fotos"])
    push!(hbox_pasta, gui.entry_pasta)
    
    push!(vbox_config, hbox_pasta)
    push!(frame_config, vbox_config)
    push!(vbox_ctrl, frame_config)
    
    # Controles da webcam
    frame_webcam = GtkFrame("Controle da Webcam")
    vbox_webcam = GtkBox(:v, 5)
    Gtk.set_gtk_property!(vbox_webcam, :margin_left, 5)
    Gtk.set_gtk_property!(vbox_webcam, :margin_right, 5)
    Gtk.set_gtk_property!(vbox_webcam, :margin_top, 5)
    Gtk.set_gtk_property!(vbox_webcam, :margin_bottom, 5)
    
    gui.btn_iniciar = GtkButton("🔴 Iniciar Webcam")
    gui.btn_parar = GtkButton("⏹️ Parar Webcam")
    Gtk.set_gtk_property!(gui.btn_parar, :sensitive, false)
    
    push!(vbox_webcam, gui.btn_iniciar)
    push!(vbox_webcam, gui.btn_parar)
    
    push!(frame_webcam, vbox_webcam)
    push!(vbox_ctrl, frame_webcam)
    
    # Captura
    frame_capture = GtkFrame("Captura de Fotos")
    vbox_capture = GtkBox(:v, 5)
    Gtk.set_gtk_property!(vbox_capture, :margin_left, 5)
    Gtk.set_gtk_property!(vbox_capture, :margin_right, 5)
    Gtk.set_gtk_property!(vbox_capture, :margin_top, 5)
    Gtk.set_gtk_property!(vbox_capture, :margin_bottom, 5)
    
    gui.btn_capturar = GtkButton("📸 Capturar Foto")
    Gtk.set_gtk_property!(gui.btn_capturar, :sensitive, false)
    push!(vbox_capture, gui.btn_capturar)
    
    push!(frame_capture, vbox_capture)
    push!(vbox_ctrl, frame_capture)
    
    # Visualização
    frame_view = GtkFrame("Visualizar")
    vbox_view = GtkBox(:v, 5)
    Gtk.set_gtk_property!(vbox_view, :margin_left, 5)
    Gtk.set_gtk_property!(vbox_view, :margin_right, 5)
    Gtk.set_gtk_property!(vbox_view, :margin_top, 5)
    Gtk.set_gtk_property!(vbox_view, :margin_bottom, 5)
    
    gui.btn_visualizar = GtkButton("🖼️ Ver Fotos")
    push!(vbox_view, gui.btn_visualizar)
    
    push!(frame_view, vbox_view)
    push!(vbox_ctrl, frame_view)
    
    push!(hbox, vbox_ctrl)
    push!(vbox, hbox)
    
    # === STATUS ===
    frame_status = GtkFrame("Status do Sistema")
    vbox_status = GtkBox(:v, 5)
    Gtk.set_gtk_property!(vbox_status, :margin_left, 5)
    Gtk.set_gtk_property!(vbox_status, :margin_right, 5)
    Gtk.set_gtk_property!(vbox_status, :margin_top, 5)
    Gtk.set_gtk_property!(vbox_status, :margin_bottom, 5)
    
    gui.label_status = GtkLabel("Sistema inicializado")
    # Remover halign - não existe em versões antigas
    push!(vbox_status, gui.label_status)
    
    gui.label_contador = GtkLabel("Fotos: 0")
    push!(vbox_status, gui.label_contador)
    
    push!(frame_status, vbox_status)
    push!(vbox, frame_status)
    
    push!(gui.window, vbox)
    
    # === CONECTAR SINAIS ===
    Gtk.signal_connect(on_iniciar_clicked, gui.btn_iniciar, "clicked")
    Gtk.signal_connect(on_parar_clicked, gui.btn_parar, "clicked")
    Gtk.signal_connect(on_capturar_clicked, gui.btn_capturar, "clicked")
    Gtk.signal_connect(on_visualizar_clicked, gui.btn_visualizar, "clicked")
    Gtk.signal_connect(on_destroy, gui.window, "destroy")
    
    gui.timer_id = nothing
end

"""
    verificar_sistema() -> Bool
Verifica se o sistema está funcionando.
"""
function verificar_sistema()
    println("🔍 Verificando sistema...")
    
    # Testar GTK
    try
        test_window = GtkWindow("Test", 100, 100)
        Gtk.destroy(test_window)
        println("✅ GTK funcionando")
    catch e
        println("❌ Erro GTK: $e")
        return false
    end
    
    # Verificar webcam (não obrigatório para iniciar)
    webcam_ok, _ = verificar_webcam()
    if webcam_ok
        println("✅ Webcam detectada")
    else
        println("⚠️ Webcam não detectada (pode ser iniciada depois)")
    end
    
    return true
end

"""
    main()
Função principal.
"""
function main()
    println("🚀 Iniciando CNN Check-In...")
    
    if !verificar_sistema()
        println("❌ Sistema não está funcionando corretamente")
        return
    end
    
    try
        criar_interface()
        Gtk.showall(gui.window)
        
        # Definir status inicial
        Gtk.set_gtk_property!(gui.label_status, :label, "Pronto - clique em 'Iniciar Webcam'")
        
        # Iniciar loop principal do GTK
        Gtk.gtk_main()
        
    catch e
        println("❌ Erro na execução: $e")
        if gui.timer_id !== nothing
            GLib.g_source_remove(gui.timer_id)
        end
        fechar_webcam()
    end
end

# Executar se for arquivo principal
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end



# === INSTRUÇÕES DE USO ===
"""
# CNN Check-In - Sistema de Captura Facial

## Pré-requisitos:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install libgtk-3-dev libcairo2-dev libpango1.0-dev v4l-utils

# Pacotes Julia
julia> using Pkg
julia> Pkg.add(["Gtk", "Cairo", "VideoIO", "Images", "FileIO", "Dates"])
```

## Execução:
```bash
julia cnncheckin_acount.jl
```

## Como usar:
1. Execute o programa
2. Configure a pasta de destino se necessário
3. Clique em "Iniciar Webcam" para ativar a câmera
4. Use "Capturar Foto" para tirar fotos individuais
5. Use "Ver Fotos" para visualizar as fotos capturadas
6. As fotos são salvas com timestamp no nome

## Funcionalidades:
- Preview em tempo real da webcam
- Captura de fotos com timestamp
- Interface GTK intuitiva
- Visualização automática das fotos
- Salvamento em formato JPEG

## Troubleshooting:
- Se a webcam não for detectada, verifique se está conectada
- Para problemas com GTK, reinstale os pacotes de desenvolvimento
- As fotos são salvas na pasta especificada (padrão: "fotos_rosto")
"""

# === INSTRUÇÕES DE USO ===
"""
# CNN Check-In - Interface GTK

## Pré-requisitos:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install libgtk-3-dev gtk2-engines-pixbuf v4l-utils
sudo apt install libcairo2-dev libpango1.0-dev

# Pacotes Julia
julia> using Pkg
julia> Pkg.add(["Gtk", "GLib", "Cairo", "VideoIO", "Images", "FileIO", "Dates"])
```

## Execução:
```bash
julia cnncheckin_gui.jl
```

## Funcionalidades:
- Preview em tempo real da webcam
- Captura individual de fotos
- Captura automática com intervalo configurável
- Visualização das fotos capturadas
- Backup automático das fotos
- Limpeza de pastas
- Interface intuitiva com GTK

## Controles:
1. Iniciar Webcam: Ativa a câmera e preview
2. Capturar Foto: Tira uma foto individual
3. Captura Automática: Sequência programada de fotos
4. Visualizar: Abre as fotos no visualizador do sistema
5. Backup: Cria cópia de segurança
6. Limpar: Remove todas as fotos (com confirmação)

## Configurações:
- Pasta: Local onde salvar as fotos
- Nº Fotos: Quantidade para captura automática
- Intervalo: Tempo entre capturas automáticas (segundos)
"""

# Para executar:
# julia cnncheckin_acount.jl

 
# pkg-config --modversion gtk+-3.0
# sudo apt update
# sudo apt install libgtk-3-dev libcairo2-dev libpango1.0-dev

# sudo apt update
# sudo apt install libgtk-3-dev libcairo2-dev libpango1.0-dev libglib2.0-dev
# sudo apt install v4l-utils  # Para webcam

# julia -e "using Pkg; Pkg.build([\"Gtk\", \"Cairo\"])"
# julia -e "using Pkg; Pkg.rm([\"Gtk\", \"GLib\", \"Cairo\"]); Pkg.add([\"Gtk\", \"GLib\", \"Cairo\"])"


# julia cnncheckin_acount.jl