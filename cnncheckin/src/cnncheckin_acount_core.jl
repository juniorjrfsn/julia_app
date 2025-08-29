# projeto: cnncheckin
# file: cnncheckin/src/cnncheckin_core.jl
# Parte 1: Funções principais e configuração do sistema

using VideoIO
using Images
using FileIO
using Dates

# Try to load optional graphics packages
graphics_available = Dict()

# Try PlotlyJS
try
    using PlotlyJS
    graphics_available["plotlyjs"] = true
catch
    graphics_available["plotlyjs"] = false
end

# Try Plots
try
    using Plots
    graphics_available["plots"] = true
catch
    graphics_available["plots"] = false
end

# Try ImageView and Gtk
try
    using ImageView, Gtk
    graphics_available["imageview"] = true
catch
    graphics_available["imageview"] = false
end

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
    
    # Testar diferentes backends gráficos em ordem de preferência
    if graphics_available["plotlyjs"]
        println("✅ PlotlyJS disponível")
        return true, "plotlyjs"
    elseif graphics_available["plots"]
        try
            Plots.gr()
            println("✅ Plots disponível")
            return true, "plots"
        catch e
            println("⚠️  Plots com erro: $(typeof(e).__name__)")
        end
    elseif graphics_available["imageview"]
        println("✅ ImageView disponível")
        return true, "imageview"
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
📐 Dimensões: $(tamanho[2])x$(tamanho[1]) pixels
💾 Tamanho: $tamanho_str
🎨 Tipo: $tipo_img$timestamp_info
📂 Caminho: $caminho_foto
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
        
        if DISPLAY_TYPE == "imageview" && graphics_available["imageview"]
            try
                imshow(img)
                return true
            catch e
                println("⚠️  Erro ImageView: $e")
            end
        elseif DISPLAY_TYPE == "plots" && graphics_available["plots"]
            try
                p = Plots.plot(img, title=titulo, axis=nothing, border=:none)
                display(p)
                return true
            catch e
                println("⚠️  Erro Plots: $e")
            end
        elseif DISPLAY_TYPE == "plotlyjs" && graphics_available["plotlyjs"]
            try
                # Converter imagem para formato PlotlyJS
                img_array = channelview(img)
                fig = PlotlyJS.plot(PlotlyJS.heatmap(z=img_array[1,:,:], colorscale="Greys"))
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
        mostrar_ascii_thumb_with_img(img, largura, basename(caminho_foto))
        return true
    catch e
        println("❌ Erro ao gerar preview ASCII: $e")
        return false
    end
end

# Função auxiliar para ASCII com imagem já carregada
function mostrar_ascii_thumb_with_img(img, largura=60, nome="Preview")
    try
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
        println("📸 $nome")
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

# Função auxiliar para preview temporário
function mostrar_imagem_temp(img)
    try
        if DISPLAY_TYPE == "imageview" && graphics_available["imageview"]
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
    mostrar_ascii_thumb_with_img(img, largura, "Preview Webcam")
end

# Função para gerenciamento de arquivos
function gerenciar_arquivos()
    println("\n🗂️  === GERENCIADOR DE ARQUIVOS ===")
    
    pastas_disponiveis = ["fotos_rosto", "fotos_rosto_simples", "fotos_rosto_preview"]
    
    while true
        println("\n📋 Opções de gerenciamento:")
        println("1 - 📊 Estatísticas das pastas")
        println("2 - 🧹 Limpeza de arquivos")
        println("3 - 📁 Criar backup")
        println("4 - 🔄 Reorganizar arquivos")
        println("5 - 🔍 Buscar por data")
        println("0 - ↩️  Voltar ao menu principal")
        
        print("\nEscolha uma opção: ")
        opcao = strip(readline())
        
        if opcao == "0"
            break
        elseif opcao == "1"
            mostrar_estatisticas(pastas_disponiveis)
        elseif opcao == "2"
            limpar_arquivos(pastas_disponiveis)
        elseif opcao == "3"
            criar_backup(pastas_disponiveis)
        elseif opcao == "4"
            reorganizar_arquivos(pastas_disponiveis)
        elseif opcao == "5"
            buscar_por_data(pastas_disponiveis)
        else
            println("❌ Opção inválida!")
        end
    end
end

function mostrar_estatisticas(pastas)
    println("\n📊 === ESTATÍSTICAS ===")
    
    total_fotos = 0
    total_tamanho = 0
    
    for pasta in pastas
        if isdir(pasta)
            fotos = listar_fotos(pasta)
            tamanho_pasta = sum(filesize(foto) for foto in fotos)
            
            println("\n📁 $pasta:")
            println("   📸 Fotos: $(length(fotos))")
            println("   💾 Tamanho: $(round(tamanho_pasta / (1024*1024), digits=2)) MB")
            
            if !isempty(fotos)
                # Primeira e última foto (por nome)
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
    println("   💾 Tamanho: $(round(total_tamanho / (1024*1024), digits=2)) MB")
end

function limpar_arquivos(pastas)
    println("\n🧹 === LIMPEZA DE ARQUIVOS ===")
    println("⚠️  ATENÇÃO: Esta operação é irreversível!")
    
    for pasta in pastas
        if isdir(pasta)
            fotos = listar_fotos(pasta)
            if !isempty(fotos)
                println("\n📁 $pasta: $(length(fotos)) fotos")
                print("Deseja limpar esta pasta? (digite 'LIMPAR' para confirmar): ")
                confirmacao = strip(readline())
                
                if confirmacao == "LIMPAR"
                    try
                        for foto in fotos
                            rm(foto)
                        end
                        println("✅ Pasta $pasta limpa!")
                    catch e
                        println("❌ Erro ao limpar $pasta: $e")
                    end
                else
                    println("❌ Operação cancelada para $pasta")
                end
            else
                println("📁 $pasta: vazia")
            end
        end
    end
end

function criar_backup(pastas)
    println("\n📁 === CRIAR BACKUP ===")
    
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
                        println("⚠️  Erro ao copiar $(basename(foto)): $e")
                    end
                end
            end
        end
    end
    
    println("✅ Backup criado: $pasta_backup")
    println("📸 Total de fotos copiadas: $total_copiadas")
end

function reorganizar_arquivos(pastas)
    println("\n🔄 === REORGANIZAR ARQUIVOS ===")
    println("Esta função renomeia arquivos seguindo padrão: foto_NNNN_data_hora.ext")
    
    for pasta in pastas
        if isdir(pasta)
            fotos = listar_fotos(pasta)
            if !isempty(fotos)
                print("Reorganizar $pasta ($(length(fotos)) fotos)? (s/N): ")
                if lowercase(strip(readline())) in ["s", "sim"]
                    println("🔄 Reorganizando $pasta...")
                    
                    for (i, foto) in enumerate(fotos)
                        try
                            timestamp = Dates.format(Dates.unix2datetime(stat(foto).mtime), "yyyy-mm-dd_HH-MM-SS")
                            _, ext = splitext(foto)
                            novo_nome = "foto_$(lpad(i, 4, '0'))_$timestamp$ext"
                            novo_caminho = joinpath(dirname(foto), novo_nome)
                            
                            if foto != novo_caminho
                                mv(foto, novo_caminho)
                                println("  ✅ $(basename(foto)) → $novo_nome")
                            end
                        catch e
                            println("  ❌ Erro com $(basename(foto)): $e")
                        end
                    end
                end
            end
        end
    end
end

function buscar_por_data(pastas)
    println("\n🔍 === BUSCAR POR DATA ===")
    print("Digite a data (formato: yyyy-mm-dd) ou parte dela: ")
    busca = strip(readline())
    
    if isempty(busca)
        println("❌ Data não informada")
        return
    end
    
    encontradas = String[]
    
    for pasta in pastas
        if isdir(pasta)
            fotos = listar_fotos(pasta)
            for foto in fotos
                if occursin(busca, basename(foto))
                    push!(encontradas, foto)
                end
            end
        end
    end
    
    if isempty(encontradas)
        println("❌ Nenhuma foto encontrada com '$busca'")
    else
        println("✅ Encontradas $(length(encontradas)) fotos:")
        for (i, foto) in enumerate(encontradas)
            tamanho_kb = round(filesize(foto) / 1024, digits=1)
            println("  $i. $(basename(foto)) ($(tamanho_kb) KB)")
        end
        
        print("\nDeseja visualizar alguma? (número ou ENTER para sair): ")
        entrada = strip(readline())
        if !isempty(entrada)
            try
                indice = parse(Int, entrada)
                if 1 <= indice <= length(encontradas)
                    mostrar_imagem(encontradas[indice])
                end
            catch
                println("❌ Número inválido")
            end
        end
    end
end