# projeto: cnncheckin
# file: cnncheckin/src/cnncheckin_acount_viewer_fixed.jl

using VideoIO
using Images
using FileIO
using Dates

# Sistema de display com múltiplos fallbacks
DISPLAY_AVAILABLE = false
DISPLAY_TYPE = "none"
global DISPLAY_AVAILABLE, DISPLAY_TYPE

# Função para detectar ambiente gráfico disponível
function detectar_ambiente_grafico()
    # Verificar se estamos em ambiente gráfico
    if !haskey(ENV, "DISPLAY") && !haskey(ENV, "WAYLAND_DISPLAY")
        println("⚠️  Nenhum ambiente gráfico detectado")
        return false, "none"
    end
    
    # Tentar diferentes backends gráficos em ordem de preferência
    backends = [
        ("PlotlyJS", () -> (using PlotlyJS; true)),
        ("Plots", () -> (using Plots; Plots.gr(); true)),
        ("ImageView", () -> (using ImageView, Gtk; true)),
    ]
    
    for (nome, teste) in backends
        try
            teste()
            println("✅ $nome disponível")
            return true, lowercase(nome)
        catch e
            println("⚠️  $nome não disponível: $(typeof(e).__name__)")
        end
    end
    
    return false, "external"
end

# Inicializar sistema de display
DISPLAY_AVAILABLE, DISPLAY_TYPE = detectar_ambiente_grafico()

if !DISPLAY_AVAILABLE
    println("🔧 Modo fallback ativo - usando visualizadores externos")
end

# Função para criar diretório se não existir
function criar_diretorio(caminho)
    if !isdir(caminho)
        mkpath(caminho)
        println("📁 Diretório criado: $caminho")
    end
end

# Função para listar todas as fotos em uma pasta
function listar_fotos(pasta)
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

# Função para exibir informações da foto
function info_foto(caminho_foto)
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
📏 Dimensões: $(tamanho[2])x$(tamanho[1]) pixels
💾 Tamanho: $tamanho_str
🎨 Tipo: $tipo_img$timestamp_info
📍 Caminho: $caminho_foto
        """
    catch e
        return "❌ Erro ao ler informações: $e"
    end
end

# Função aprimorada para visualizar imagens
function mostrar_imagem(caminho_foto, titulo="Imagem")
    if !isfile(caminho_foto)
        println("❌ Arquivo não encontrado: $caminho_foto")
        return false
    end
    
    try
        img = load(caminho_foto)
        
        if DISPLAY_TYPE == "imageview"
            try
                imshow(img)
                return true
            catch e
                println("⚠️  Erro ImageView: $e")
            end
        elseif DISPLAY_TYPE == "plots"
            try
                using Plots
                p = plot(img, title=titulo, axis=nothing, border=:none)
                display(p)
                return true
            catch e
                println("⚠️  Erro Plots: $e")
            end
        elseif DISPLAY_TYPE == "plotlyjs"
            try
                using PlotlyJS
                # Converter imagem para formato PlotlyJS
                img_array = channelview(img)
                fig = plot(heatmap(z=img_array[1,:,:], colorscale="Greys"))
                display(fig)
                return true
            catch e
                println("⚠️  Erro PlotlyJS: $e")
            end
        end
        
        # Fallback para visualizador externo
        return abrir_com_visualizador_externo(caminho_foto)
        
    catch e
        println("❌ Erro ao carregar imagem: $e")
        return false
    end
end

# Função melhorada para visualizador externo
function abrir_com_visualizador_externo(caminho_foto)
    if !isfile(caminho_foto)
        println("❌ Arquivo não encontrado")
        return false
    end
    
    visualizadores = []
    
    if Sys.islinux()
        # Lista expandida de visualizadores Linux
        linux_viewers = [
            ("eog", "Eye of GNOME"),
            ("gwenview", "KDE Gwenview"),
            ("feh", "Feh (leve)"),
            ("display", "ImageMagick"),
            ("gimp", "GIMP"),
            ("firefox", "Firefox"),
            ("google-chrome", "Google Chrome"),
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
                # Verificar se o comando existe
                run(pipeline(`which $comando`, devnull), wait=true)
                
                if comando == "xdg-open"
                    run(`$comando $caminho_foto`, wait=false)
                elseif comando in ["firefox", "google-chrome"]
                    run(`$comando file://$caminho_foto`, wait=false)
                else
                    run(`$comando $caminho_foto`, wait=false)
                end
            elseif Sys.iswindows()
                run(`cmd /c start "" "$caminho_foto"`, wait=false)
            elseif Sys.isapple()
                run(`open $caminho_foto`, wait=false)
            end
            
            println("🖼️  Imagem aberta com $nome")
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

# Função para criar thumbnail em ASCII (fallback criativo)
function mostrar_ascii_thumb(caminho_foto, largura=60)
    try
        img = load(caminho_foto)
        
        # Redimensionar para ASCII
        h, w = size(img)
        nova_h = round(Int, largura * h / w / 2)  # /2 para compensar proporção dos caracteres
        
        # Converter para escala de cinza
        if eltype(img) <: RGB
            img_gray = Gray.(img)
        else
            img_gray = img
        end
        
        # Redimensionar
        img_small = imresize(img_gray, (nova_h, largura))
        
        # Caracteres ASCII por intensidade
        chars = " .:-=+*#%@"
        
        println("\n" * "="^largura)
        println("📸 Preview ASCII: $(basename(caminho_foto))")
        println("="^largura)
        
        for i in 1:size(img_small, 1)
            linha = ""
            for j in 1:size(img_small, 2)
                intensidade = gray(img_small[i, j])
                char_idx = min(length(chars), max(1, round(Int, intensidade * length(chars))))
                linha *= chars[char_idx]
            end
            println(linha)
        end
        println("="^largura)
        
        return true
    catch e
        println("❌ Erro ao gerar preview ASCII: $e")
        return false
    end
end

# Visualizador de fotos aprimorado
function visualizar_fotos()
    pastas_disponiveis = ["fotos_rosto", "fotos_rosto_simples", "fotos_rosto_preview"]
    pasta_escolhida = nothing
    
    println("🖼️  === VISUALIZADOR DE FOTOS CAPTURADAS ===")
    println("Sistema gráfico: $(DISPLAY_TYPE)")
    println("\nPastas disponíveis:")
    
    pastas_existentes = []
    for (i, pasta) in enumerate(pastas_disponiveis)
        if isdir(pasta)
            fotos = listar_fotos(pasta)
            println("$i - $pasta ($(length(fotos)) fotos)")
            push!(pastas_existentes, (i, pasta))
        else
            println("$i - $pasta (pasta não existe)")
        end
    end
    
    if isempty(pastas_existentes)
        println("❌ Nenhuma pasta com fotos encontrada!")
        println("💡 Execute primeiro a captura de fotos")
        return
    end
    
    print("\nEscolha a pasta (1-$(length(pastas_disponiveis))): ")
    escolha = readline()
    
    try
        indice = parse(Int, escolha)
        if 1 <= indice <= length(pastas_disponiveis)
            pasta_escolhida = pastas_disponiveis[indice]
        else
            println("❌ Escolha inválida!")
            return
        end
    catch
        println("❌ Entrada inválida!")
        return
    end
    
    fotos = listar_fotos(pasta_escolhida)
    
    if isempty(fotos)
        println("❌ Nenhuma foto encontrada na pasta $pasta_escolhida")
        return
    end
    
    println("\n🖼️  Encontradas $(length(fotos)) fotos!")
    println("\n📋 Comandos disponíveis:")
    println("─────────────────────────")
    println("• ENTER ou 'n': próxima foto")
    println("• 'p': foto anterior")
    println("• 'i': informações detalhadas")
    println("• 'o': abrir com visualizador externo")
    println("• 'a': preview ASCII (texto)")
    println("• 'l': listar todas as fotos")
    println("• 'j <número>': pular para foto específica")
    println("• 'd': deletar foto atual")
    println("• 'r': renomear foto atual")
    println("• 'q': sair")
    println("─────────────────────────")
    
    foto_atual = 1
    
    while true
        if foto_atual < 1
            foto_atual = 1
        elseif foto_atual > length(fotos)
            foto_atual = length(fotos)
        end
        
        caminho_foto = fotos[foto_atual]
        
        println("\n" * "="^80)
        println("📸 Foto $foto_atual de $(length(fotos))")
        println("📁 $(basename(caminho_foto))")
        println("🗂️  Pasta: $(dirname(caminho_foto))")
        println("="^80)
        
        # Tentar mostrar a imagem
        sucesso_display = false
        if DISPLAY_AVAILABLE || DISPLAY_TYPE == "external"
            sucesso_display = mostrar_imagem(caminho_foto, "Foto $foto_atual")
        end
        
        if !sucesso_display
            println("💡 Use 'o' para visualizador externo ou 'a' para preview ASCII")
        end
        
        print("\n[Foto $foto_atual/$(length(fotos))] Comando: ")
        comando = lowercase(strip(readline()))
        
        if comando in ["q", "sair", "exit"]
            break
            
        elseif comando in ["", "n", "next"]
            foto_atual += 1
            if foto_atual > length(fotos)
                println("📚 Última foto alcançada! Voltando para a primeira...")
                foto_atual = 1
            end
            
        elseif comando in ["p", "prev", "anterior"]
            foto_atual -= 1
            if foto_atual < 1
                println("🔙 Primeira foto alcançada! Indo para a última...")
                foto_atual = length(fotos)
            end
            
        elseif comando in ["i", "info"]
            println(info_foto(fotos[foto_atual]))
            
        elseif comando in ["o", "open"]
            abrir_com_visualizador_externo(fotos[foto_atual])
            
        elseif comando in ["a", "ascii"]
            mostrar_ascii_thumb(fotos[foto_atual])
            
        elseif comando in ["d", "delete"]
            println("⚠️  ATENÇÃO: Deletar foto permanentemente!")
            print("Tem certeza que deseja deletar '$(basename(fotos[foto_atual]))'? (digite 'DELETAR'): ")
            confirmacao = strip(readline())
            if confirmacao == "DELETAR"
                try
                    rm(fotos[foto_atual])
                    println("✅ Foto deletada permanentemente!")
                    deleteat!(fotos, foto_atual)
                    if foto_atual > length(fotos) && !isempty(fotos)
                        foto_atual = length(fotos)
                    elseif isempty(fotos)
                        println("🗑️  Todas as fotos foram deletadas!")
                        break
                    end
                catch e
                    println("❌ Erro ao deletar: $e")
                end
            else
                println("❌ Operação cancelada")
            end
            
        elseif comando in ["r", "rename"]
            print("📝 Novo nome (sem extensão): ")
            novo_nome = strip(readline())
            if !isempty(novo_nome)
                try
                    _, ext = splitext(fotos[foto_atual])
                    novo_caminho = joinpath(dirname(fotos[foto_atual]), novo_nome * ext)
                    mv(fotos[foto_atual], novo_caminho)
                    fotos[foto_atual] = novo_caminho
                    println("✅ Foto renomeada para: $(basename(novo_caminho))")
                catch e
                    println("❌ Erro ao renomear: $e")
                end
            end
            
        elseif comando in ["l", "list"]
            println("\n📋 Lista completa de fotos:")
            println("─" * "─"^60)
            for (i, foto) in enumerate(fotos)
                marcador = i == foto_atual ? "➤ " : "  "
                tamanho = round(filesize(foto) / 1024, digits=1)
                println("$marcador$(lpad(i, 3)). $(basename(foto)) ($(tamanho) KB)")
            end
            println("─" * "─"^60)
            
        elseif startswith(comando, "j ") || startswith(comando, "jump ")
            try
                numero_str = split(comando, r"\s+", limit=2)[2]
                numero = parse(Int, numero_str)
                if 1 <= numero <= length(fotos)
                    foto_atual = numero
                    println("🔄 Pulando para foto $numero")
                else
                    println("❌ Número inválido! Use 1-$(length(fotos))")
                end
            catch
                println("❌ Formato inválido! Use: j <número>")
            end
            
        else
            println("❌ Comando não reconhecido: '$comando'")
            println("💡 Digite 'q' para sair ou veja os comandos disponíveis acima")
        end
    end
    
    println("\n👋 Visualizador encerrado!")
end

# Função aprimorada para verificar webcam
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
                    println("📏 Resolução: $(size(frame))")
                end
                
                close(camera)
                return true, i
            catch e
                if i == 0
                    println("⚠️  Webcam padrão (índice 0): $(typeof(e).__name__)")
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

# Função aprimorada para capturar fotos
function capturar_fotos_rosto()
    println("🔍 Verificando sistema...")
    
    webcam_ok, camera_index = verificar_webcam()
    if !webcam_ok
        println("\n🔧 Soluções possíveis:")
        println("   1. Verifique se a webcam está conectada")
        println("   2. Feche outros programas que podem estar usando a webcam")
        println("   3. Verifique permissões: sudo usermod -a -G video $USER")
        println("   4. Reinicie o sistema se necessário")
        return
    end
    
    # Configurações personalizáveis
    pasta_fotos = "fotos_rosto"
    
    println("\n📸 === CAPTURADOR DE FOTOS FACIAIS ===")
    print("Quantas fotos capturar? (padrão: 10): ")
    input_fotos = strip(readline())
    num_fotos = isempty(input_fotos) ? 10 : parse(Int, input_fotos)
    
    print("Intervalo entre fotos em segundos? (padrão: 3): ")
    input_intervalo = strip(readline())
    intervalo = isempty(input_intervalo) ? 3 : parse(Int, input_intervalo)
    
    criar_diretorio(pasta_fotos)
    
    println("\n📋 Configuração:")
    println("   📁 Pasta: $pasta_fotos")
    println("   📸 Fotos: $num_fotos")
    println("   ⏱️  Intervalo: $intervalo segundos")
    println("   📷 Câmera: índice $camera_index")
    
    println("\n🎯 Instruções:")
    println("   • Posicione-se em frente à webcam")
    println("   • Mude de ângulo e expressão a cada captura")
    println("   • Mantenha boa iluminação")
    println("   • Pressione ENTER para iniciar")
    
    readline()
    
    try
        camera = VideoIO.opencamera(camera_index)
        println("🔴 Webcam iniciada! Preparando...")
        
        # Warm-up da câmera
        for _ in 1:3
            try
                read(camera)
                sleep(0.5)
            catch
                break
            end
        end
        
        # Countdown inicial
        for i in 5:-1:1
            print("\r⏳ Primeira foto em $i segundos... ")
            flush(stdout)
            sleep(1)
        end
        println("\n🚀 Iniciando capturas!")
        
        fotos_capturadas = String[]
        
        for foto_num in 1:num_fotos
            try
                # Capturar múltiplos frames e pegar o melhor
                melhor_frame = nothing
                for tentativa in 1:3
                    frame = read(camera)
                    if frame !== nothing
                        melhor_frame = frame
                        break
                    end
                    sleep(0.1)
                end
                
                if melhor_frame !== nothing
                    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS-sss")
                    nome_arquivo = "foto_$(lpad(foto_num, 2, '0'))_$timestamp.jpg"
                    caminho_completo = joinpath(pasta_fotos, nome_arquivo)
                    
                    save(caminho_completo, melhor_frame)
                    push!(fotos_capturadas, caminho_completo)
                    
                    tamanho_kb = round(filesize(caminho_completo) / 1024, digits=1)
                    println("✅ Foto $foto_num/$num_fotos: $nome_arquivo ($(tamanho_kb) KB)")
                    
                    # Tentar mostrar preview
                    if DISPLAY_AVAILABLE
                        mostrar_imagem(caminho_completo, "Foto $foto_num capturada")
                    end
                    
                    # Countdown para próxima foto
                    if foto_num < num_fotos
                        println("🔄 Próxima foto em $intervalo segundos... Mude de posição!")
                        for i in intervalo:-1:1
                            print("\r⏰ Próxima captura em $i segundos... ")
                            flush(stdout)
                            sleep(1)
                        end
                        println()
                    end
                else
                    println("❌ Erro ao capturar frame $foto_num")
                end
            catch e
                println("❌ Erro durante captura $foto_num: $e")
            end
        end
        
        close(camera)
        
        println("\n🎉 CAPTURA CONCLUÍDA!")
        println("📊 Estatísticas:")
        println("   ✅ Fotos capturadas: $(length(fotos_capturadas))/$num_fotos")
        println("   📁 Pasta: $pasta_fotos")
        
        if !isempty(fotos_capturadas)
            tamanho_total = sum(filesize(foto) for foto in fotos_capturadas)
            println("   💾 Tamanho total: $(round(tamanho_total / (1024*1024), digits=2)) MB")
            
            print("\n🖼️  Deseja visualizar as fotos agora? (s/N): ")
            if lowercase(strip(readline())) in ["s", "sim", "y", "yes"]
                println()
                visualizar_fotos()
            end
        end
        
    catch e
        println("❌ Erro crítico: $e")
    end
end

# Função de captura manual aprimorada
function capturar_fotos_simples()
    println("🔍 Verificando webcam...")
    webcam_ok, camera_index = verificar_webcam()
    if !webcam_ok
        return
    end
    
    pasta_fotos = "fotos_rosto_simples"
    criar_diretorio(pasta_fotos)
    
    println("\n📷 === MODO CAPTURA MANUAL ===")
    println("Comandos:")
    println("  • ENTER: capturar foto")
    println("  • 'info': informações da webcam")
    println("  • 'preview': mostrar frame atual")
    println("  • 'sair' ou 'q': terminar")
    
    try
        camera = VideoIO.opencamera(camera_index)
        foto_count = 0
        
        # Warm-up
        for _ in 1:3
            try
                read(camera)
                sleep(0.2)
            catch
                break
            end
        end
        
        println("\n🔴 Webcam ativa! Pronto para capturar.")
        
        while true
            print("\n[$(foto_count) fotos] Comando: ")
            entrada = lowercase(strip(readline()))
            
            if entrada in ["sair", "q", "quit", "exit"]
                break
            elseif entrada == "info"
                try
                    frame = read(camera)
                    if frame !== nothing
                        println("📊 Info da webcam:")
                        println("   📏 Resolução: $(size(frame))")
                        println("   🎨 Tipo: $(typeof(frame))")
                        println("   📷 Índice: $camera_index")
                    end
                catch e
                    println("❌ Erro ao obter info: $e")
                end
            elseif entrada == "preview"
                try
                    frame = read(camera)
                    if frame !== nothing
                        if DISPLAY_AVAILABLE
                            mostrar_imagem_temp(frame)
                        else
                            mostrar_ascii_thumb_temp(frame)
                        end
                    end
                catch e
                    println("❌ Erro no preview: $e")
                end
            elseif entrada == "" || entrada == "capturar"
                try
                    # Capturar com múltiplas tentativas
                    frame_capturado = nothing
                    for _ in 1:5
                        frame = read(camera)
                        if frame !== nothing
                            frame_capturado = frame
                            break
                        end
                        sleep(0.1)
                    end
                    
                    if frame_capturado !== nothing
                        timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS-sss")
                        nome_arquivo = "manual_$(lpad(foto_count + 1, 3, '0'))_$timestamp.jpg"
                        caminho_completo = joinpath(pasta_fotos, nome_arquivo)
                        
                        save(caminho_completo, frame_capturado)
                        foto_count += 1
                        
                        tamanho_kb = round(filesize(caminho_completo) / 1024, digits=1)
                        println("✅ Foto $foto_count salva: $nome_arquivo ($(tamanho_kb) KB)")
                        
                        # Preview da foto capturada
                        if DISPLAY_AVAILABLE
                            mostrar_imagem(caminho_completo, "Foto $foto_count")
                        end
                    else
                        println("❌ Falha ao capturar frame")
                    end
                catch e
                    println("❌ Erro na captura: $e")
                end
            else
                println("❌ Comando não reconhecido: '$entrada'")
            end
        end
        
        close(camera)
        
        println("\n🎉 Captura manual finalizada!")
        if foto_count > 0
            println("📊 Total: $foto_count fotos salvas em: $pasta_fotos")
        else
            println("📷 Nenhuma foto capturada")
        end
        
    catch e
        println("❌ Erro ao acessar webcam: $e")
    end
end

# Função auxiliar para preview temporário
function mostrar_imagem_temp(img)
    try
        if DISPLAY_TYPE == "imageview"
            imshow(img)
        else
            println("🖼️  Preview carregado (feche a janela para continuar)")
        end
        return true
    catch e
        return mostrar_ascii_thumb_temp(img)
    end
end

function mostrar_ascii_thumb_temp(img, largura=40)
    println("\n📺 Preview ASCII:")
    mostrar_ascii_thumb("", largura, img)
end

# Menu principal aprimorado
function main()
    println("🔴 === CNN CHECK-IN - SISTEMA DE CAPTURA FACIAL ===")
    println("🖥️  Sistema: $(Sys.KERNEL) $(Sys.ARCH)")
    println("🎨 Display: $DISPLAY_TYPE")
    
    if haskey(ENV, "DISPLAY")
        println("📺 DISPLAY: $(ENV["DISPLAY"])")
    end
    
    while true
        println("\n" * "="^50)
        println("📋 MENU PRINCIPAL")
        println("="^50)
        println("1 - 📸 Captura Automática")
        println("2 - 🖱️  Captura Manual")
        println("3 - 🖼️  Visualizar Fotos")
        println("4 - 🔧 Diagnóstico do Sistema")
        println("5 - 🗑️  Limpar Fotos Antigas")
        println("6 - 📊 Estatísticas das Fotos")
        println("7 - ❌ Sair")
        println("="^50)
        
        print("🎯 Escolha uma opção (1-7): ")
        escolha = strip(readline())
        
        if escolha == "1"
            capturar_fotos_rosto()
        elseif escolha == "2"
            capturar_fotos_simples()
        elseif escolha == "3"
            visualizar_fotos()
        elseif escolha == "4"
            diagnostico_sistema()
        elseif escolha == "5"
            limpar_fotos_antigas()
        elseif escolha == "6"
            estatisticas_fotos()
        elseif escolha == "7"
            println("👋 Até logo!")
            break
        else
            println("❌ Escolha inválida! Digite um número de 1 a 7")
        end
        
        # Pausa antes de mostrar o menu novamente
        println("\n⏸️  Pressione ENTER para continuar...")
        readline()
    end
end

# Função de diagnóstico do sistema
function diagnostico_sistema()
    println("\n🔧 === DIAGNÓSTICO DO SISTEMA ===")
    
    # Informações básicas
    println("📊 Sistema Operacional:")
    println("   • Kernel: $(Sys.KERNEL)")
    println("   • Arquitetura: $(Sys.ARCH)")
    println("   • Versão Julia: $(VERSION)")
    
    # Variáveis de ambiente importantes
    println("\n🌐 Ambiente Gráfico:")
    env_vars = ["DISPLAY", "WAYLAND_DISPLAY", "XDG_SESSION_TYPE", "XDG_CURRENT_DESKTOP"]
    for var in env_vars
        valor = haskey(ENV, var) ? ENV[var] : "não definida"
        println("   • $var: $valor")
    end
    
    # Status dos pacotes
    println("\n📦 Pacotes Julia:")
    pacotes_graficos = ["ImageView", "Gtk", "Plots", "PlotlyJS"]
    for pkg in pacotes_graficos
        try
            eval(Meta.parse("using $pkg"))
            println("   ✅ $pkg: disponível")
        catch e
            println("   ❌ $pkg: $(typeof(e).__name__)")
        end
    end
    
    # Verificar webcam
    println("\n📷 Webcam:")
    webcam_ok, camera_index = verificar_webcam()
    if webcam_ok
        println("   ✅ Webcam funcionando (índice: $camera_index)")
    else
        println("   ❌ Webcam não detectada")
    end
    
    # Verificar visualizadores externos
    println("\n🖼️  Visualizadores de Imagem:")
    if Sys.islinux()
        viewers = ["eog", "gwenview", "feh", "display", "firefox", "xdg-open"]
        for viewer in viewers
            try
                run(pipeline(`which $viewer`, devnull), wait=true)
                println("   ✅ $viewer: disponível")
            catch
                println("   ❌ $viewer: não encontrado")
            end
        end
    else
        println("   ℹ️  Sistema não-Linux detectado")
    end
    
    # Verificar permissões
    println("\n🔒 Permissões:")
    user = ENV["USER"]
    try
        grupos = split(read(`groups $user`, String))
        if "video" in grupos
            println("   ✅ Usuário no grupo 'video'")
        else
            println("   ⚠️  Usuário NÃO está no grupo 'video'")
            println("      Execute: sudo usermod -a -G video $user")
        end
    catch
        println("   ❓ Não foi possível verificar grupos")
    end
    
    # Sugestões de melhorias
    println("\n💡 Sugestões para melhorar o sistema:")
    
    if !DISPLAY_AVAILABLE
        println("   📺 Para interface gráfica:")
        println("      sudo apt install libgtk-3-dev gtk2-engines-pixbuf")
        println("      export DISPLAY=:0.0")
    end
    
    if Sys.islinux()
        println("   🖼️  Para visualizadores de imagem:")
        println("      sudo apt install eog gwenview feh imagemagick")
    end
    
    println("   📷 Para problemas de webcam:")
    println("      sudo apt install v4l-utils")
    println("      v4l2-ctl --list-devices")
end

# Função para limpar fotos antigas
function limpar_fotos_antigas()
    println("\n🗑️  === LIMPEZA DE FOTOS ANTIGAS ===")
    
    pastas = ["fotos_rosto", "fotos_rosto_simples", "fotos_rosto_preview"]
    total_fotos = 0
    total_tamanho = 0
    
    # Análise inicial
    for pasta in pastas
        if isdir(pasta)
            fotos = listar_fotos(pasta)
            tamanho_pasta = sum(filesize(foto) for foto in fotos)
            total_fotos += length(fotos)
            total_tamanho += tamanho_pasta
            
            println("📁 $pasta: $(length(fotos)) fotos ($(round(tamanho_pasta/(1024*1024), digits=2)) MB)")
        end
    end
    
    if total_fotos == 0
        println("✅ Nenhuma foto encontrada para limpar!")
        return
    end
    
    println("\n📊 Total: $total_fotos fotos ($(round(total_tamanho/(1024*1024), digits=2)) MB)")
    
    println("\nOpções de limpeza:")
    println("1 - 🗑️  Deletar todas as fotos")
    println("2 - 📅 Deletar fotos mais antigas que X dias")
    println("3 - 📁 Escolher pasta específica")
    println("4 - ❌ Cancelar")
    
    print("Escolha: ")
    opcao = strip(readline())
    
    if opcao == "1"
        print("⚠️  ATENÇÃO: Isso deletará TODAS as fotos! Digite 'DELETAR TUDO' para confirmar: ")
        if strip(readline()) == "DELETAR TUDO"
            fotos_deletadas = 0
            for pasta in pastas
                if isdir(pasta)
                    fotos = listar_fotos(pasta)
                    for foto in fotos
                        try
                            rm(foto)
                            fotos_deletadas += 1
                        catch e
                            println("❌ Erro ao deletar $(basename(foto)): $e")
                        end
                    end
                end
            end
            println("✅ $fotos_deletadas fotos deletadas!")
        else
            println("❌ Operação cancelada")
        end
        
    elseif opcao == "2"
        print("Deletar fotos mais antigas que quantos dias? ")
        try
            dias = parse(Int, strip(readline()))
            data_limite = now() - Dates.Day(dias)
            fotos_deletadas = 0
            
            for pasta in pastas
                if isdir(pasta)
                    fotos = listar_fotos(pasta)
                    for foto in fotos
                        try
                            data_arquivo = Dates.unix2datetime(stat(foto).mtime)
                            if data_arquivo < data_limite
                                rm(foto)
                                fotos_deletadas += 1
                                println("🗑️  Deletado: $(basename(foto))")
                            end
                        catch e
                            println("❌ Erro com $(basename(foto)): $e")
                        end
                    end
                end
            end
            
            println("✅ $fotos_deletadas fotos antigas deletadas!")
            
        catch
            println("❌ Número de dias inválido!")
        end
        
    elseif opcao == "3"
        println("Escolha a pasta:")
        for (i, pasta) in enumerate(pastas)
            if isdir(pasta)
                fotos = listar_fotos(pasta)
                println("$i - $pasta ($(length(fotos)) fotos)")
            else
                println("$i - $pasta (vazia)")
            end
        end
        
        print("Pasta: ")
        try
            indice = parse(Int, strip(readline()))
            if 1 <= indice <= length(pastas)
                pasta_escolhida = pastas[indice]
                if isdir(pasta_escolhida)
                    fotos = listar_fotos(pasta_escolhida)
                    print("Deletar $(length(fotos)) fotos de $pasta_escolhida? (digite 'SIM'): ")
                    if strip(readline()) == "SIM"
                        fotos_deletadas = 0
                        for foto in fotos
                            try
                                rm(foto)
                                fotos_deletadas += 1
                            catch e
                                println("❌ Erro: $e")
                            end
                        end
                        println("✅ $fotos_deletadas fotos deletadas de $pasta_escolhida!")
                    else
                        println("❌ Operação cancelada")
                    end
                end
            end
        catch
            println("❌ Opção inválida!")
        end
    end
end

# Função para mostrar estatísticas
function estatisticas_fotos()
    println("\n📊 === ESTATÍSTICAS DAS FOTOS ===")
    
    pastas = ["fotos_rosto", "fotos_rosto_simples", "fotos_rosto_preview"]
    estatisticas_globais = Dict(
        "total_fotos" => 0,
        "total_tamanho" => 0,
        "maior_foto" => ("", 0),
        "menor_foto" => ("", Inf),
        "foto_mais_antiga" => ("", now()),
        "foto_mais_recente" => ("", DateTime(1900))
    )
    
    for pasta in pastas
        if !isdir(pasta)
            continue
        end
        
        fotos = listar_fotos(pasta)
        if isempty(fotos)
            println("📁 $pasta: vazia")
            continue
        end
        
        # Estatísticas da pasta
        tamanho_pasta = 0
        tamanhos = Float64[]
        datas = DateTime[]
        
        for foto in fotos
            try
                tamanho = filesize(foto)
                tamanho_pasta += tamanho
                push!(tamanhos, tamanho)
                
                # Data de modificação
                data = Dates.unix2datetime(stat(foto).mtime)
                push!(datas, data)
                
                # Atualizar estatísticas globais
                estatisticas_globais["total_fotos"] += 1
                estatisticas_globais["total_tamanho"] += tamanho
                
                if tamanho > estatisticas_globais["maior_foto"][2]
                    estatisticas_globais["maior_foto"] = (foto, tamanho)
                end
                
                if tamanho < estatisticas_globais["menor_foto"][2]
                    estatisticas_globais["menor_foto"] = (foto, tamanho)
                end
                
                if data < estatisticas_globais["foto_mais_antiga"][2]
                    estatisticas_globais["foto_mais_antiga"] = (foto, data)
                end
                
                if data > estatisticas_globais["foto_mais_recente"][2]
                    estatisticas_globais["foto_mais_recente"] = (foto, data)
                end
                
            catch e
                println("⚠️  Erro ao analisar $(basename(foto)): $e")
            end
        end
        
        # Exibir estatísticas da pasta
        println("\n📁 Pasta: $pasta")
        println("   📸 Fotos: $(length(fotos))")
        println("   💾 Tamanho total: $(round(tamanho_pasta/(1024*1024), digits=2)) MB")
        println("   📏 Tamanho médio: $(round(mean(tamanhos)/1024, digits=1)) KB")
        println("   📈 Maior foto: $(round(maximum(tamanhos)/1024, digits=1)) KB")
        println("   📉 Menor foto: $(round(minimum(tamanhos)/1024, digits=1)) KB")
        
        if !isempty(datas)
            println("   📅 Mais antiga: $(Dates.format(minimum(datas), "dd/mm/yyyy HH:MM"))")
            println("   📅 Mais recente: $(Dates.format(maximum(datas), "dd/mm/yyyy HH:MM"))")
        end
    end
    
    # Estatísticas globais
    if estatisticas_globais["total_fotos"] > 0
        println("\n" * "="^50)
        println("📊 ESTATÍSTICAS GLOBAIS")
        println("="^50)
        println("📸 Total de fotos: $(estatisticas_globais["total_fotos"])")
        println("💾 Espaço ocupado: $(round(estatisticas_globais["total_tamanho"]/(1024*1024), digits=2)) MB")
        
        maior_nome = basename(estatisticas_globais["maior_foto"][1])
        maior_tamanho = round(estatisticas_globais["maior_foto"][2]/1024, digits=1)
        println("📈 Maior foto: $maior_nome ($(maior_tamanho) KB)")
        
        menor_nome = basename(estatisticas_globais["menor_foto"][1])
        menor_tamanho = round(estatisticas_globais["menor_foto"][2]/1024, digits=1)
        println("📉 Menor foto: $menor_nome ($(menor_tamanho) KB)")
        
        data_antiga = Dates.format(estatisticas_globais["foto_mais_antiga"][2], "dd/mm/yyyy HH:MM")
        println("🕰️  Mais antiga: $(basename(estatisticas_globais["foto_mais_antiga"][1])) ($data_antiga)")
        
        data_recente = Dates.format(estatisticas_globais["foto_mais_recente"][2], "dd/mm/yyyy HH:MM")
        println("🆕 Mais recente: $(basename(estatisticas_globais["foto_mais_recente"][1])) ($data_recente)")
        
        # Cálculo de média de uso de espaço por dia
        if estatisticas_globais["foto_mais_antiga"][2] != estatisticas_globais["foto_mais_recente"][2]
            dias_diferenca = Dates.value(estatisticas_globais["foto_mais_recente"][2] - estatisticas_globais["foto_mais_antiga"][2]) / (1000 * 60 * 60 * 24)
            if dias_diferenca > 0
                fotos_por_dia = round(estatisticas_globais["total_fotos"] / dias_diferenca, digits=2)
                mb_por_dia = round(estatisticas_globais["total_tamanho"] / (1024*1024) / dias_diferenca, digits=2)
                println("📈 Média: $fotos_por_dia fotos/dia ($(mb_por_dia) MB/dia)")
            end
        end
        
    else
        println("\n📭 Nenhuma foto encontrada no sistema!")
    end
end

# Função auxiliar para mostrar ASCII thumb com imagem direta
function mostrar_ascii_thumb(caminho_foto, largura=60, img_direta=nothing)
    try
        img = img_direta !== nothing ? img_direta : load(caminho_foto)
        
        # Redimensionar para ASCII
        h, w = size(img)
        nova_h = max(1, round(Int, largura * h / w / 2))
        
        # Converter para escala de cinza
        if eltype(img) <: RGB
            img_gray = Gray.(img)
        else
            img_gray = img
        end
        
        # Redimensionar
        img_small = imresize(img_gray, (nova_h, largura))
        
        # Caracteres ASCII por intensidade (mais detalhados)
        chars = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@\$"
        
        if img_direta === nothing
            println("📸 Preview ASCII: $(basename(caminho_foto))")
        else
            println("📸 Preview ASCII em tempo real:")
        end
        println("─" * repeat("─", largura-1))
        
        for i in 1:size(img_small, 1)
            linha = ""
            for j in 1:size(img_small, 2)
                intensidade = gray(img_small[i, j])
                char_idx = min(length(chars), max(1, round(Int, intensidade * length(chars))))
                linha *= chars[char_idx]
            end
            println(linha)
        end
        println("─" * repeat("─", largura-1))
        
        return true
    catch e
        println("❌ Erro ao gerar preview ASCII: $e")
        return false
    end
end

# Executar apenas se for o arquivo principal
if abspath(PROGRAM_FILE) == @__FILE__
    try
        main()
    catch InterruptException
        println("\n\n👋 Programa interrompido pelo usuário. Até logo!")
    catch e
        println("\n❌ Erro inesperado: $e")
        println("📧 Reporte este erro para suporte técnico")
    end
end

# Instruções de uso no final do arquivo
"""
# === INSTRUÇÕES DE USO ===

## Instalação das dependências:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install libgtk-3-dev gtk2-engines-pixbuf v4l-utils
sudo apt install eog gwenview feh imagemagick  # visualizadores

# Adicionar usuário ao grupo video
sudo usermod -a -G video \$USER
# Logout e login novamente

# Variáveis de ambiente (se necessário)
export DISPLAY=:0.0
```

## Pacotes Julia necessários:
```julia
using Pkg
Pkg.add(["VideoIO", "Images", "FileIO", "Dates"])

# Opcionais para interface gráfica
Pkg.add(["ImageView", "Gtk", "Plots", "PlotlyJS"])
```

## Execução:
```bash
julia cnncheckin_acount_viewer_fixed.jl
```

## Solução de problemas comuns:

### Webcam não detectada:
- Verifique conexão: `lsusb | grep -i camera`
- Liste dispositivos: `v4l2-ctl --list-devices`
- Teste com: `ffmpeg -f v4l2 -list_formats all -i /dev/video0`

### Problemas gráficos:
- Tente: `xhost +local:`
- Verifique: `echo \$DISPLAY`
- Teste: `xclock` ou `xeyes`

### Permissões:
- Grupo video: `groups \$USER | grep video`
- Permissões dev: `ls -l /dev/video*`
"""


#   # Salve como cnncheckin_acount_viewer_fixed.jl
#  julia cnncheckin_acount_viewer_fixed.jl


# # Instalar dependências GTK
# sudo apt update
# sudo apt install libgtk-3-dev libgtk-3-0

# # Se ainda houver problemas, tente:
# sudo apt install gtk2-engines-pixbuf
# export DISPLAY=:0.0

# # Ou instale um visualizador de imagens
# sudo apt install eog  # Eye of GNOME
# # ou
# sudo apt install gwenview  # KDE
# # ou  
# sudo apt install feh  # Leve e rápido