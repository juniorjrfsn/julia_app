#!/usr/bin/env julia
# projeto: cnncheckin
# file: cnncheckin/src/cnncheckin_acount.jl  
# Versão corrigida com compatibilidade GTK4 e sintaxe atualizada
 

println("🚀 Iniciando CNN Check-In...")

# Carregar dependências
using VideoIO, Images, FileIO, Dates

# Tentar carregar GTK4
try
    using Gtk4
    println("✅ GTK4 carregado com sucesso")
catch e
    println("📦 Instalando GTK4...")
    using Pkg
    Pkg.add("Gtk4")
    using Gtk4
    println("✅ GTK4 instalado e carregado")
end

# Estado global simplificado
mutable struct SimpleApp
    camera
    active::Bool
    folder::String
    count::Int
    
    SimpleApp() = new(nothing, false, "fotos_rosto", 0)
end

const app = SimpleApp()

"""Criar diretório se não existir"""
create_dir(path) = !isdir(path) && mkpath(path)

"""Verificar se webcam está disponível"""
function check_camera()
    println("🔍 Verificando webcam...")
    for i in 0:5
        try
            cam = VideoIO.opencamera(i)
            frame = read(cam)
            close(cam)
            if frame !== nothing && size(frame, 1) > 10 && size(frame, 2) > 10
                println("✅ Webcam encontrada no índice $i")
                return i
            end
        catch
            continue
        end
    end
    println("❌ Nenhuma webcam funcionando encontrada")
    return -1
end

"""Inicializar webcam"""
function start_camera()
    if app.active
        return true
    end
    
    cam_idx = check_camera()
    if cam_idx < 0
        return false
    end
    
    try
        app.camera = VideoIO.opencamera(cam_idx)
        app.active = true
        println("📹 Webcam ativada")
        return true
    catch e
        println("❌ Erro ao ativar webcam: $e")
        return false
    end
end

"""Parar webcam"""
function stop_camera()
    if app.active && app.camera !== nothing
        try
            close(app.camera)
            app.active = false
            app.camera = nothing
            println("ℹ️ Webcam desativada")
        catch e
            println("⚠️ Erro ao desativar webcam: $e")
        end
    end
end

"""Capturar uma foto"""
function capture_photo(folder_name=app.folder)
    if !app.active || app.camera === nothing
        return false, "Webcam não está ativa"
    end
    
    try
        frame = read(app.camera)
        if frame === nothing
            return false, "Erro ao ler frame da webcam"
        end
        
        # Criar pasta
        create_dir(folder_name)
        app.folder = folder_name
        
        # Nome único com timestamp
        timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS-sss")
        filename = "foto_$(app.count + 1)_$timestamp.jpg"
        filepath = joinpath(folder_name, filename)
        
        # Salvar imagem
        save(filepath, frame)
        app.count += 1
        
        println("📸 Foto $filename salva com sucesso!")
        return true, "Foto salva: $filename"
        
    catch e
        return false, "Erro ao capturar foto: $e"
    end
end

"""Abrir pasta no explorador"""
function open_folder(folder_name=app.folder)
    if !isdir(folder_name)
        return "Pasta $folder_name não existe"
    end
    
    try
        if Sys.islinux()
            run(`xdg-open $folder_name`, wait=false)
        elseif Sys.iswindows()
            run(`explorer $folder_name`, wait=false)
        elseif Sys.isapple()
            run(`open $folder_name`, wait=false)
        end
        return "Pasta aberta: $folder_name"
    catch
        return "Erro ao abrir pasta"
    end
end

"""Interface gráfica"""
function create_gui()
    # Janela principal
    win = GtkWindow("CNN Check-In - Captura Facial", 550, 350)
    
    # Container principal
    vbox = GtkBox(:v, 10)
    set_margin_start(vbox, 10)
    set_margin_end(vbox, 10)
    set_margin_top(vbox, 10)
    set_margin_bottom(vbox, 10)
    
    # Título - CORREÇÃO AQUI: usar GtkLabel com texto
    title = GtkLabel("CNN Check-In")
    set_markup(title, "<span size='x-large' weight='bold'>CNN Check-In</span>")
    set_halign(title, Gtk4.Align_CENTER)
    
    # Campo para nome da pasta
    hbox1 = GtkBox(:h, 5)
    set_homogeneous(hbox1, false)
    append(hbox1, GtkLabel("Pasta:"))
    entry_folder = GtkEntry()
    set_text(entry_folder, "fotos_rosto")
    set_hexpand(entry_folder, true)
    append(hbox1, entry_folder)
    
    # Botões de controle da webcam
    hbox2 = GtkBox(:h, 5)
    set_homogeneous(hbox2, true)
    btn_start = GtkButton("🔴 Iniciar Webcam")
    btn_stop = GtkButton("ℹ️ Parar Webcam")
    set_sensitive(btn_stop, false)
    append(hbox2, btn_start)
    append(hbox2, btn_stop)
    
    # Botões de ação
    hbox3 = GtkBox(:h, 5)
    set_homogeneous(hbox3, true)
    btn_capture = GtkButton("📸 Capturar Foto")
    btn_open = GtkButton("📁 Abrir Pasta")
    set_sensitive(btn_capture, false)
    append(hbox3, btn_capture)
    append(hbox3, btn_open)
    
    # Labels de status
    lbl_status = GtkLabel("Sistema inicializado")
    lbl_count = GtkLabel("Fotos: 0")
    set_halign(lbl_status, Gtk4.Align_START)
    set_halign(lbl_count, Gtk4.Align_START)
    
    # Montar interface
    for widget in [title, hbox1, hbox2, hbox3, lbl_status, lbl_count]
        append(vbox, widget)
    end
    
    set_child(win, vbox)
    
    # Callbacks
    signal_connect(btn_start, "clicked") do button
        if start_camera()
            set_sensitive(btn_start, false)
            set_sensitive(btn_stop, true)
            set_sensitive(btn_capture, true)
            set_text(lbl_status, "✅ Webcam ativa")
        else
            set_text(lbl_status, "❌ Erro ao iniciar webcam")
        end
    end
    
    signal_connect(btn_stop, "clicked") do button
        stop_camera()
        set_sensitive(btn_start, true)
        set_sensitive(btn_stop, false)
        set_sensitive(btn_capture, false)
        set_text(lbl_status, "ℹ️ Webcam parada")
    end
    
    signal_connect(btn_capture, "clicked") do button
        folder = get_text(entry_folder)
        success, msg = capture_photo(folder)
        set_text(lbl_status, msg)
        set_text(lbl_count, "Fotos: $(app.count)")
    end
    
    signal_connect(btn_open, "clicked") do button
        folder = get_text(entry_folder)
        msg = open_folder(folder)
        set_text(lbl_status, msg)
    end
    
    signal_connect(win, "close-request") do widget
        stop_camera()
        exit()
    end
    
    return (win, lbl_status)
end

"""Função principal"""
function main()
    try
        println("🔧 Criando interface...")
        
        window, status_label = create_gui()
        
        # Verificar webcam inicial
        cam_idx = check_camera()
        if cam_idx >= 0
            set_text(status_label, "✅ Webcam detectada - Clique 'Iniciar'")
        else
            set_text(status_label, "⚠️ Webcam não detectada")
        end
        
        # Mostrar janela
        show(window)
        
        println("✅ Interface criada com sucesso!")
        println("\n📋 INSTRUÇÕES:")
        println("1. Clique 'Iniciar Webcam' para ativar a câmera")
        println("2. Use 'Capturar Foto' para tirar fotos")
        println("3. 'Abrir Pasta' mostra onde as fotos estão salvas")
        println("4. Feche a janela para sair")
        
        return window
        
    catch e
        println("❌ Erro ao criar interface: $e")
        println("Stack trace:")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
        stop_camera()
        return nothing
    end
end

# Executar se for chamado diretamente
if abspath(PROGRAM_FILE) == @__FILE__
    global gui_window = main()
    
    # Manter programa rodando se executado como script
    if gui_window !== nothing
        try
            # Loop principal GTK4
            app_gtk = GtkApplication("com.cnn.checkin", 0)
            
            signal_connect(app_gtk, "activate") do app
                # Já temos a janela criada, apenas precisamos mantê-la
                nothing
            end
            
            # Executar aplicação GTK
            run(app_gtk, String[])
            
        catch InterruptException
            println("\n👋 Saindo...")
        finally
            stop_camera()
        end
    end
end

println("\n" * "="^60)
println("🎯 CNN CHECK-IN - SISTEMA DE CAPTURA FACIAL")
println("="^60)
println("📝 Para executar:")
println("   julia cnncheckin_acount.jl")
println("\n🔧 Ou no REPL Julia:")
println("   julia> include(\"cnncheckin_acount.jl\")")
println("   julia> main()")
println("="^60)

# === INSTRUÇÕES DE USO ===
"""
# CNN Check-In - Sistema de Captura Facial

## Pré-requisitos:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install libgtk-4-dev libcairo2-dev libpango1.0-dev v4l-utils

# Pacotes Julia
julia> using Pkg
julia> Pkg.add(["Gtk4", "Cairo", "VideoIO", "Images", "FileIO", "Dates"])

 
## Como usar:
1. Execute o programa
2. Configure a pasta de destino se necessário
3. Clique em "Iniciar Webcam" para ativar a câmera
4. Use "Capturar Foto" para tirar fotos individuais
5. Use "Ver Fotos" para visualizar as fotos capturadas
6. As fotos são salvas com timestamp no nome

 

## Funcionalidades:
- Captura de fotos com timestamp
- Interface GTK4 moderna
- Visualização automática das fotos
- Salvamento em formato JPEG
- Preview em tempo real da webcam
- Interface intuitiva e responsiva

## Troubleshooting:
- Se a webcam não for detectada, verifique se está conectada
- Para problemas com GTK4, reinstale os pacotes de desenvolvimento
- As fotos são salvas na pasta especificada (padrão: "fotos_rosto")
- Os avisos do GTK sobre módulos podem ser ignorados


 

## Controles:
1. Iniciar Webcam: Ativa a câmera e preview
2. Capturar Foto: Tira uma foto individual
3. Visualizar: Abre as fotos no visualizador do sistema
4. Parar Webcam: Desliga a câmera

## Configurações:
- Pasta: Local onde salvar as fotos
"""

# === INSTRUÇÕES DE USO ===
"""
# CNN Check-In - Sistema de Captura Facial

## Pré-requisitos:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install libgtk-3-dev libcairo2-dev libpango1.0-dev v4l-utils


```

## Execução:
```bash
julia cnncheckin_acount.jl
```

## Controles:
1. Iniciar Webcam: Ativa a câmera e preview
2. Capturar Foto: Tira uma foto individual
3. Captura Automática: Sequência programada de fotos
4. Visualizar: Abre as fotos no visualizador do sistema
5. Backup: Cria cópia de segurança
6. Limpar: Remove todas as fotos (com confirmação)

 
 
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


```


"""

# Para executar:
# julia cnncheckin_acount.jl

 
# pkg-config --modversion gtk+-3.0
# sudo apt update
# sudo apt install libgtk-3-dev libcairo2-dev libpango1.0-dev

# sudo apt update
# sudo apt install libgtk-3-dev libcairo2-dev libpango1.0-dev libglib2.0-dev
# sudo apt install v4l-utils  # Para webcam

 


# julia cnncheckin_acount.jl