# projeto: cnncheckin
# file: cnncheckin/src/cnncheckin_acount.jl


# projeto: cnncheckin
# file: cnncheckin/src/cnncheckin_viewer.jl

using VideoIO
using ImageView
using Images
using FileIO
using Dates
using Gtk

# Função para criar diretório se não existir
function criar_diretorio(caminho)
    if !isdir(caminho)
        mkpath(caminho)
        println("Diretório criado: $caminho")
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
        return "Arquivo não encontrado"
    end
    
    try
        img = load(caminho_foto)
        nome_arquivo = basename(caminho_foto)
        tamanho = size(img)
        
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
        
        return "📁 Arquivo: $nome_arquivo\n📏 Dimensões: $(tamanho[2])x$(tamanho[1]) pixels$timestamp_info"
    catch e
        return "Erro ao ler informações: $e"
    end
end

# Visualizador de fotos com navegação
function visualizar_fotos()
    pastas_disponiveis = ["fotos_rosto", "fotos_rosto_simples"]
    pasta_escolhida = nothing
    
    println("=== VISUALIZADOR DE FOTOS CAPTURADAS ===")
    println("Pastas disponíveis:")
    
    for (i, pasta) in enumerate(pastas_disponiveis)
        if isdir(pasta)
            fotos = listar_fotos(pasta)
            println("$i - $pasta ($(length(fotos)) fotos)")
        else
            println("$i - $pasta (pasta não existe)")
        end
    end
    
    print("Escolha a pasta (1-$(length(pastas_disponiveis))): ")
    escolha = readline()
    
    try
        indice = parse(Int, escolha)
        if 1 <= indice <= length(pastas_disponiveis)
            pasta_escolhida = pastas_disponiveis[indice]
        else
            println("Escolha inválida!")
            return
        end
    catch
        println("Entrada inválida!")
        return
    end
    
    fotos = listar_fotos(pasta_escolhida)
    
    if isempty(fotos)
        println("Nenhuma foto encontrada na pasta $pasta_escolhida")
        return
    end
    
    println("\n🖼️  Encontradas $(length(fotos)) fotos!")
    println("Comandos:")
    println("- ENTER ou 'n': próxima foto")
    println("- 'p': foto anterior")
    println("- 'i': informações da foto atual")
    println("- 'l': listar todas as fotos")
    println("- 'j <número>': pular para foto específica")
    println("- 'q': sair")
    
    foto_atual = 1
    
    while true
        if foto_atual < 1
            foto_atual = 1
        elseif foto_atual > length(fotos)
            foto_atual = length(fotos)
        end
        
        caminho_foto = fotos[foto_atual]
        
        try
            img = load(caminho_foto)
            
            # Exibir a imagem
            println("\n" * "="^50)
            println("📸 Foto $foto_atual de $(length(fotos))")
            println("📁 $(basename(caminho_foto))")
            println("="^50)
            
            # Tentar exibir a imagem
            try
                imshow(img)
            catch e
                println("⚠️  Não foi possível exibir a imagem na janela: $e")
                println("Mas a imagem existe e pode ser aberta manualmente.")
            end
            
        catch e
            println("❌ Erro ao carregar imagem: $e")
        end
        
        print("\n[Foto $foto_atual/$(length(fotos))] Comando: ")
        comando = lowercase(strip(readline()))
        
        if comando == "q" || comando == "sair"
            break
        elseif comando == "" || comando == "n" || comando == "next"
            foto_atual += 1
            if foto_atual > length(fotos)
                println("📍 Última foto alcançada!")
                foto_atual = length(fotos)
            end
        elseif comando == "p" || comando == "prev" || comando == "anterior"
            foto_atual -= 1
            if foto_atual < 1
                println("📍 Primeira foto alcançada!")
                foto_atual = 1
            end
        elseif comando == "i" || comando == "info"
            println("\n" * info_foto(fotos[foto_atual]))
        elseif comando == "l" || comando == "list"
            println("\n📋 Lista de todas as fotos:")
            for (i, foto) in enumerate(fotos)
                marcador = i == foto_atual ? "➤ " : "  "
                println("$marcador$i. $(basename(foto))")
            end
        elseif startswith(comando, "j ") || startswith(comando, "jump ")
            try
                numero = parse(Int, split(comando)[2])
                if 1 <= numero <= length(fotos)
                    foto_atual = numero
                    println("🔄 Pulando para foto $numero")
                else
                    println("❌ Número inválido! Use 1-$(length(fotos))")
                end
            catch
                println("❌ Formato inválido! Use 'j <número>'")
            end
        else
            println("❌ Comando não reconhecido!")
        end
    end
    
    println("\n👋 Visualizador encerrado!")
end

# Função para capturar fotos com preview em tempo real
function capturar_fotos_com_preview()
    pasta_fotos = "fotos_rosto_preview"
    criar_diretorio(pasta_fotos)
    
    println("=== CAPTURADOR COM PREVIEW ===")
    println("Esta função mostra a webcam em tempo real")
    println("Pressione ESPAÇO para capturar uma foto")
    println("Pressione 'q' para sair")
    println("Pressione ENTER para começar...")
    readline()
    
    try
        camera = VideoIO.opencamera()
        foto_count = 0
        
        println("📹 Webcam iniciada! Pressione ESPAÇO para capturar, 'q' para sair")
        
        # Loop principal de captura com preview
        while true
            try
                frame = read(camera)
                if frame !== nothing
                    # Mostrar preview contínuo
                    try
                        imshow(frame)
                    catch
                        # Se não conseguir mostrar, continuar
                    end
                    
                    # Verificar entrada do usuário (simulação - em uma implementação real
                    # seria necessário capturar eventos de teclado de forma não-bloqueante)
                    print("\r📷 Foto $foto_count capturadas | ESPAÇO=capturar, ENTER=continuar, 'q'+ENTER=sair: ")
                    
                    # Aguardar entrada
                    entrada = readline()
                    entrada = strip(entrada)
                    
                    if lowercase(entrada) == "q"
                        break
                    elseif entrada == " " || lowercase(entrada) == "capturar"
                        # Capturar foto
                        timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
                        nome_arquivo = "foto_preview_$(foto_count + 1)_$timestamp.jpg"
                        caminho_completo = joinpath(pasta_fotos, nome_arquivo)
                        
                        save(caminho_completo, frame)
                        foto_count += 1
                        
                        println("✅ Foto $foto_count salva: $nome_arquivo")
                        sleep(1)  # Pause para mostrar a mensagem
                    end
                else
                    println("❌ Erro ao capturar frame")
                    break
                end
                
            catch e
                println("❌ Erro durante captura: $e")
                break
            end
        end
        
        close(camera)
        println("\n🎉 Captura finalizada! $foto_count fotos salvas em $pasta_fotos")
        
    catch e
        println("❌ Erro ao acessar webcam: $e")
    end
end

# Função para capturar fotos do rosto (mantida do código original)
function capturar_fotos_rosto()
    # Configurações
    pasta_fotos = "fotos_rosto"
    num_fotos = 10  # Número de fotos a capturar
    intervalo = 3   # Intervalo em segundos entre capturas
    
    # Criar diretório para as fotos
    criar_diretorio(pasta_fotos)
    
    println("=== CAPTURADOR DE FOTOS FACIAIS ===")
    println("Instruções:")
    println("- Posicione-se em frente à webcam")
    println("- Mude de ângulo a cada captura (frontal, perfil esquerdo, perfil direito, etc.)")
    println("- Pressione ENTER para iniciar")
    println("- Pressione 'q' na janela da webcam para sair antecipadamente")
    println()
    
    readline()  # Aguarda pressionar ENTER
    
    try
        # Abrir webcam (geralmente índice 0 para webcam padrão)
        camera = VideoIO.opencamera()
        
        println("📹 Webcam iniciada! Preparando para capturar $num_fotos fotos...")
        println("⏰ Primeira foto em 5 segundos...")
        
        # Aguardar 5 segundos antes da primeira captura
        for i in 5:-1:1
            print("\r⏳ $i segundos... ")
            sleep(1)
        end
        println("\n🚀 Iniciando capturas!")
        
        foto_count = 0
        
        while foto_count < num_fotos
            try
                # Capturar frame da webcam
                frame = read(camera)
                
                if frame !== nothing
                    # Gerar nome único para a foto
                    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
                    nome_arquivo = "foto_$(foto_count + 1)_$timestamp.jpg"
                    caminho_completo = joinpath(pasta_fotos, nome_arquivo)
                    
                    # Salvar a imagem
                    save(caminho_completo, frame)
                    
                    foto_count += 1
                    
                    println("✅ Foto $foto_count/$num_fotos salva: $nome_arquivo")
                    
                    if foto_count < num_fotos
                        println("📐 Próxima foto em $intervalo segundos... Mude de ângulo!")
                        
                        # Mostrar preview da imagem capturada
                        try
                            imshow(frame)
                            for i in intervalo:-1:1
                                print("\r⏰ Próxima captura em $i segundos... ")
                                sleep(1)
                            end
                            println()
                        catch
                            # Se imshow não funcionar, apenas aguardar
                            sleep(intervalo)
                        end
                    end
                else
                    println("❌ Erro ao capturar frame da webcam")
                    break
                end
                
            catch e
                println("❌ Erro durante captura: $e")
                break
            end
        end
        
        # Fechar webcam
        close(camera)
        
        if foto_count == num_fotos
            println("\n🎉 Captura concluída com sucesso!")
            println("📁 $foto_count fotos salvas na pasta: $pasta_fotos")
        else
            println("\n⚠️ Captura interrompida. $foto_count fotos salvas.")
        end
        
    catch e
        println("❌ Erro ao acessar webcam: $e")
        println("\n💡 Dicas para resolver:")
        println("- Verifique se a webcam está conectada")
        println("- Feche outros programas que possam estar usando a webcam")
        println("- Execute o script com permissões adequadas")
    end
end

# Função alternativa com interface mais simples
function capturar_fotos_simples()
    pasta_fotos = "fotos_rosto_simples"
    criar_diretorio(pasta_fotos)
    
    println("=== MODO SIMPLES ===")
    println("Pressione ENTER para cada captura, ou digite 'sair' para terminar")
    
    try
        camera = VideoIO.opencamera()
        foto_count = 0
        
        while true
            println("\n📸 Posicione-se e pressione ENTER para capturar (ou 'sair'):")
            entrada = readline()
            
            if lowercase(strip(entrada)) == "sair"
                break
            end
            
            try
                frame = read(camera)
                if frame !== nothing
                    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
                    nome_arquivo = "foto_$(foto_count + 1)_$timestamp.jpg"
                    caminho_completo = joinpath(pasta_fotos, nome_arquivo)
                    
                    save(caminho_completo, frame)
                    foto_count += 1
                    
                    println("✅ Foto salva: $nome_arquivo")
                    
                    # Mostrar preview
                    try
                        imshow(frame)
                        sleep(2)
                    catch
                        # Continuar se não conseguir mostrar
                    end
                else
                    println("❌ Erro ao capturar frame")
                end
            catch e
                println("❌ Erro na captura: $e")
            end
        end
        
        close(camera)
        println("\n🎉 $foto_count fotos salvas na pasta: $pasta_fotos")
        
    catch e
        println("❌ Erro ao acessar webcam: $e")
    end
end

# Menu principal melhorado
function main()
    println("🔴 === CNN CHECK-IN - SISTEMA DE CAPTURA E VISUALIZAÇÃO ===")
    println()
    println("Escolha uma opção:")
    println("1 - 📷 Captura Automática (10 fotos com intervalo)")
    println("2 - 🖱️  Captura Manual (pressione ENTER para cada foto)")
    println("3 - 👁️  Captura com Preview em Tempo Real")
    println("4 - 🖼️  Visualizar Fotos Capturadas")
    println("5 - ❌ Sair")
    print("\n🔵 Escolha (1-5): ")
    
    escolha = readline()
    
    if escolha == "1"
        capturar_fotos_rosto()
    elseif escolha == "2"
        capturar_fotos_simples()
    elseif escolha == "3"
        capturar_fotos_com_preview()
    elseif escolha == "4"
        visualizar_fotos()
    elseif escolha == "5"
        println("👋 Até logo!")
        return
    else
        println("❌ Escolha inválida!")
        println()
        main()
    end
    
    # Perguntar se quer fazer algo mais
    println("\n🔄 Deseja fazer algo mais?")
    println("1 - Sim, voltar ao menu")
    println("2 - Não, sair")
    print("Escolha: ")
    
    continuar = readline()
    if continuar == "1"
        println()
        main()
    else
        println("👋 Até logo!")
    end
end

# Executar o programa
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

# Para executar:
# julia cnncheckin_acount.jl

 