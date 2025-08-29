# projeto: cnncheckin
# file: cnncheckin/src/cnncheckin_acount_viewer.jl
# Parte 2: Visualizador, captura e interface principal

# Incluir o arquivo principal (assumindo que está no mesmo diretório)
include("cnncheckin_core.jl")

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
    println("─"^50)
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
    println("─"^50)
    
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
        println("📂 $(basename(caminho_foto))")
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
                println("📙 Primeira foto alcançada! Indo para a última...")
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

# Função aprimorada para capturar fotos automaticamente
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
    println("  • 'config': alterar configurações")
    println("  • 'sair' ou 'q': terminar")
    
    # Configurações padrão
    qualidade_jpg = 95
    resolucao_personalizada = nothing
    
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
        println("💡 Dica: Use 'preview' para ver o enquadramento atual")
        
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
                        println("   ⚙️  Qualidade JPG: $qualidade_jpg%")
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
                
            elseif entrada == "config"
                println("\n⚙️  Configurações:")
                print("Nova qualidade JPG (atual: $qualidade_jpg, 1-100): ")
                input_qual = strip(readline())
                if !isempty(input_qual)
                    try
                        nova_qual = parse(Int, input_qual)
                        if 1 <= nova_qual <= 100
                            qualidade_jpg = nova_qual
                            println("✅ Qualidade alterada para: $qualidade_jpg%")
                        else
                            println("❌ Qualidade deve ser entre 1-100")
                        end
                    catch
                        println("❌ Valor inválido")
                    end
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
                        
                        # Salvar com qualidade especificada (simulado - VideoIO pode não suportar)
                        save(caminho_completo, frame_capturado)
                        foto_count += 1
                        
                        tamanho_kb = round(filesize(caminho_completo) / 1024, digits=1)
                        println("✅ Foto $foto_count salva: $nome_arquivo ($(tamanho_kb) KB)")
                        
                        # Preview da foto capturada
                        if DISPLAY_AVAILABLE
                            mostrar_imagem(caminho_completo, "Foto $foto_count")
                        else
                            println("💡 Use 'preview' para ver o resultado ou visualizador externo depois")
                        end
                    else
                        println("❌ Falha ao capturar frame")
                    end
                catch e
                    println("❌ Erro na captura: $e")
                end
                
            elseif entrada == "help" || entrada == "ajuda"
                println("\n📋 Comandos disponíveis:")
                println("  ENTER    - Capturar foto")
                println("  preview  - Mostrar preview da webcam")
                println("  info     - Informações da webcam")
                println("  config   - Alterar configurações")
                println("  help     - Mostrar esta ajuda")
                println("  q/sair   - Sair do programa")
                
            else
                println("❌ Comando não reconhecido: '$entrada'")
                println("💡 Digite 'help' para ver comandos disponíveis")
            end
        end
        
        close(camera)
        
        println("\n🎉 Captura manual finalizada!")
        if foto_count > 0
            println("📊 Total: $foto_count fotos salvas em: $pasta_fotos")
            
            print("🖼️  Deseja visualizar as fotos? (s/N): ")
            if lowercase(strip(readline())) in ["s", "sim", "y", "yes"]
                visualizar_fotos()
            end
        else
            println("📷 Nenhuma foto capturada")
        end
        
    catch e
        println("❌ Erro ao acessar webcam: $e")
    end
end

# Função para captura com timer/delay personalizável
function capturar_com_timer()
    println("🔍 Verificando webcam...")
    webcam_ok, camera_index = verificar_webcam()
    if !webcam_ok
        return
    end
    
    pasta_fotos = "fotos_rosto_timer"
    criar_diretorio(pasta_fotos)
    
    println("\n⏰ === CAPTURA COM TIMER PERSONALIZADO ===")
    
    print("Quantas fotos? (padrão: 5): ")
    input_fotos = strip(readline())
    num_fotos = isempty(input_fotos) ? 5 : parse(Int, input_fotos)
    
    print("Timer para cada foto (segundos, padrão: 10): ")
    input_timer = strip(readline())
    timer_segundos = isempty(input_timer) ? 10 : parse(Int, input_timer)
    
    print("Intervalo entre fotos (segundos, padrão: 2): ")
    input_intervalo = strip(readline())
    intervalo = isempty(input_intervalo) ? 2 : parse(Int, input_intervalo)
    
    println("\n📋 Configuração do Timer:")
    println("   📸 Fotos: $num_fotos")
    println("   ⏰ Timer por foto: $timer_segundos segundos")
    println("   ⏱️  Intervalo entre fotos: $intervalo segundos")
    println("   📁 Pasta: $pasta_fotos")
    
    print("\nPressione ENTER para iniciar...")
    readline()
    
    try
        camera = VideoIO.opencamera(camera_index)
        println("🔴 Webcam iniciada!")
        
        # Warm-up
        for _ in 1:3
            read(camera)
            sleep(0.2)
        end
        
        fotos_capturadas = String[]
        
        for foto_num in 1:num_fotos
            println("\n📸 === PREPARANDO FOTO $foto_num/$num_fotos ===")
            
            # Preview antes do timer
            try
                frame_preview = read(camera)
                if frame_preview !== nothing && DISPLAY_AVAILABLE
                    mostrar_imagem_temp(frame_preview)
                end
            catch
                # Ignore preview errors
            end
            
            # Countdown do timer
            println("⏰ Timer iniciado para foto $foto_num:")
            for i in timer_segundos:-1:1
                if i <= 5
                    print("\r🔥 CAPTURA EM $i... ")
                else
                    print("\r⏰ Captura em $i segundos... ")
                end
                flush(stdout)
                sleep(1)
            end
            
            # Capturar foto
            print("\r📸 CAPTURANDO! ")
            flush(stdout)
            
            try
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
                    nome_arquivo = "timer_$(lpad(foto_num, 2, '0'))_$timestamp.jpg"
                    caminho_completo = joinpath(pasta_fotos, nome_arquivo)
                    
                    save(caminho_completo, melhor_frame)
                    push!(fotos_capturadas, caminho_completo)
                    
                    tamanho_kb = round(filesize(caminho_completo) / 1024, digits=1)
                    println("✅ CAPTURADA! ($nome_arquivo - $(tamanho_kb) KB)")
                    
                    # Mostrar foto capturada
                    if DISPLAY_AVAILABLE
                        mostrar_imagem(caminho_completo, "Foto $foto_num - Timer")
                    end
                else
                    println("❌ Falha na captura!")
                end
            catch e
                println("❌ Erro durante captura: $e")
            end
            
            # Intervalo para próxima foto
            if foto_num < num_fotos
                println("⏸️  Intervalo de $intervalo segundos...")
                sleep(intervalo)
            end
        end
        
        close(camera)
        
        println("\n🎉 CAPTURA COM TIMER CONCLUÍDA!")
        println("📊 Resumo:")
        println("   ✅ Fotos capturadas: $(length(fotos_capturadas))/$num_fotos")
        println("   📁 Pasta: $pasta_fotos")
        
        if !isempty(fotos_capturadas)
            tamanho_total = sum(filesize(foto) for foto in fotos_capturadas)
            println("   💾 Tamanho total: $(round(tamanho_total / (1024*1024), digits=2)) MB")
        end
        
    catch e
        println("❌ Erro durante captura com timer: $e")
    end
end

# Menu principal completo
function main()
    println("🔴 === CNN CHECK-IN - SISTEMA DE CAPTURA FACIAL ===")
    println("🖥️  Sistema: $(Sys.KERNEL) $(Sys.ARCH)")
    println("🎨 Display: $DISPLAY_TYPE")
    
    if haskey(ENV, "DISPLAY")
        println("📺 DISPLAY: $(ENV["DISPLAY"])")
    end
    
    while true
        println("\n" * "="^60)
        println("📋 MENU PRINCIPAL")
        println("="^60)
        println("1 - 📸 Captura Automática (série com intervalo)")
        println("2 - 🖱️  Captura Manual (sob demanda)")
        println("3 - ⏰ Captura com Timer (countdown personalizado)")
        println("4 - 🖼️  Visualizar Fotos")
        println("5 - 🗂️  Gerenciar Arquivos")
        println("6 - 🔍 Testar Webcam")
        println("7 - ℹ️  Informações do Sistema")
        println("8 - 🛠️  Configurações")
        println("0 - 🚪 Sair")
        println("="^60)
        
        print("Escolha uma opção (0-8): ")
        opcao = strip(readline())
        
        if opcao == "0"
            println("👋 Até logo!")
            break
            
        elseif opcao == "1"
            capturar_fotos_rosto()
            
        elseif opcao == "2"
            capturar_fotos_simples()
            
        elseif opcao == "3"
            capturar_com_timer()
            
        elseif opcao == "4"
            visualizar_fotos()
            
        elseif opcao == "5"
            gerenciar_arquivos()
            
        elseif opcao == "6"
            println("\n🔍 === TESTE DE WEBCAM ===")
            webcam_ok, camera_index = verificar_webcam()
            if webcam_ok
                println("✅ Webcam funcionando corretamente!")
                print("Deseja fazer um teste de captura? (s/N): ")
                if lowercase(strip(readline())) in ["s", "sim"]
                    try
                        camera = VideoIO.opencamera(camera_index)
                        frame = read(camera)
                        if frame !== nothing
                            println("✅ Teste de captura bem-sucedido!")
                            if DISPLAY_AVAILABLE
                                mostrar_imagem_temp(frame)
                            else
                                mostrar_ascii_thumb_temp(frame)
                            end
                        end
                        close(camera)
                    catch e
                        println("❌ Erro no teste: $e")
                    end
                end
            else
                println("❌ Problema com a webcam detectado")
            end
            
        elseif opcao == "7"
            mostrar_info_sistema()
            
        elseif opcao == "8"
            menu_configuracoes()
            
        else
            println("❌ Opção inválida! Escolha entre 0-8.")
        end
        
        if opcao != "0"
            print("\nPressione ENTER para continuar...")
            readline()
        end
    end
end

# Função para mostrar informações do sistema
function mostrar_info_sistema()
    println("\n💻 === INFORMAÇÕES DO SISTEMA ===")
    println("🖥️  Sistema Operacional: $(Sys.KERNEL)")
    println("🏗️  Arquitetura: $(Sys.ARCH)")
    println("📦 Versão Julia: $(VERSION)")
    println("🎨 Sistema Gráfico: $DISPLAY_TYPE")
    
    # Informações de ambiente
    if haskey(ENV, "DISPLAY")
        println("📺 DISPLAY: $(ENV["DISPLAY"])")
    end
    if haskey(ENV, "WAYLAND_DISPLAY")
        println("🪟 WAYLAND_DISPLAY: $(ENV["WAYLAND_DISPLAY"])")
    end
    
    # Status dos pacotes gráficos
    println("\n📚 Pacotes Gráficos Disponíveis:")
    for (pkg, disponivel) in graphics_available
        status = disponivel ? "✅" : "❌"
        println("   $status $pkg")
    end
    
    # Informações das pastas
    println("\n📁 Status das Pastas de Fotos:")
    pastas = ["fotos_rosto", "fotos_rosto_simples", "fotos_rosto_timer", "fotos_rosto_preview"]
    for pasta in pastas
        if isdir(pasta)
            fotos = listar_fotos(pasta)
            tamanho = sum(filesize(foto) for foto in fotos) / (1024*1024)
            println("   📂 $pasta: $(length(fotos)) fotos ($(round(tamanho, digits=2)) MB)")
        else
            println("   📁 $pasta: não existe")
        end
    end
    
    # Testar webcam rapidamente
    println("\n📷 Status da Webcam:")
    webcam_ok, camera_index = verificar_webcam()
    if webcam_ok
        println("   ✅ Webcam disponível (índice: $camera_index)")
    else
        println("   ❌ Nenhuma webcam detectada")
    end
end

# Menu de configurações
function menu_configuracoes()
    println("\n⚙️  === CONFIGURAÇÕES ===")
    
    while true
        println("\n📋 Opções de Configuração:")
        println("1 - 🎨 Preferências de Display")
        println("2 - 📁 Gerenciar Pastas Padrão")
        println("3 - 🔧 Configurações de Captura")
        println("4 - 🧹 Limpeza Geral")
        println("5 - 📊 Exportar Relatório")
        println("0 - ↩️  Voltar")
        
        print("\nEscolha: ")
        opcao = strip(readline())
        
        if opcao == "0"
            break
        elseif opcao == "1"
            config_display()
        elseif opcao == "2"
            config_pastas()
        elseif opcao == "3"
            config_captura()
        elseif opcao == "4"
            limpeza_geral()
        elseif opcao == "5"
            exportar_relatorio()
        else
            println("❌ Opção inválida!")
        end
    end
end

function config_display()
    println("\n🎨 === CONFIGURAÇÕES DE DISPLAY ===")
    println("Sistema atual: $DISPLAY_TYPE")
    println("Display disponível: $DISPLAY_AVAILABLE")
    
    println("\nTentando redetectar sistema gráfico...")
    global DISPLAY_AVAILABLE, DISPLAY_TYPE
    DISPLAY_AVAILABLE, DISPLAY_TYPE = detectar_ambiente_grafico()
    
    println("Novo status: $DISPLAY_TYPE")
end

function config_pastas()
    println("\n📁 === GERENCIAR PASTAS PADRÃO ===")
    pastas_padrao = ["fotos_rosto", "fotos_rosto_simples", "fotos_rosto_timer", "fotos_rosto_preview", "backup_fotos"]
    
    for pasta in pastas_padrao
        print("📂 Criar pasta '$pasta'? (s/N): ")
        if lowercase(strip(readline())) in ["s", "sim"]
            criar_diretorio(pasta)
        end
    end
end

function config_captura()
    println("\n🔧 === CONFIGURAÇÕES DE CAPTURA ===")
    println("Configurações são definidas durante cada sessão de captura.")
    println("💡 Dicas:")
    println("   • Use boa iluminação")
    println("   • Posicione a câmera na altura dos olhos")
    println("   • Mantenha distância de 50-80cm da câmera")
    println("   • Evite fundos complexos")
end

function limpeza_geral()
    println("\n🧹 === LIMPEZA GERAL ===")
    println("⚠️  Esta operação removerá TODAS as fotos e backups!")
    print("Digite 'LIMPAR TUDO' para confirmar: ")
    
    if strip(readline()) == "LIMPAR TUDO"
        pastas = ["fotos_rosto", "fotos_rosto_simples", "fotos_rosto_timer", "fotos_rosto_preview"]
        total_removidas = 0
        
        for pasta in pastas
            if isdir(pasta)
                fotos = listar_fotos(pasta)
                for foto in fotos
                    try
                        rm(foto)
                        total_removidas += 1
                    catch e
                        println("⚠️  Erro ao remover $(basename(foto)): $e")
                    end
                end
                
                # Remover pasta se vazia
                try
                    if isempty(readdir(pasta))
                        rm(pasta)
                        println("🗑️  Pasta $pasta removida")
                    end
                catch
                    # Ignore errors
                end
            end
        end
        
        # Limpar backups
        for item in readdir(".")
            if startswith(item, "backup_fotos_") && isdir(item)
                print("🗑️  Remover backup '$item'? (s/N): ")
                if lowercase(strip(readline())) in ["s", "sim"]
                    try
                        rm(item, recursive=true)
                        println("✅ Backup removido: $item")
                    catch e
                        println("❌ Erro ao remover backup: $e")
                    end
                end
            end
        end
        
        println("✅ Limpeza concluída! $total_removidas fotos removidas.")
    else
        println("❌ Operação cancelada")
    end
end

function exportar_relatorio()
    println("\n📊 === EXPORTAR RELATÓRIO ===")
    
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    arquivo_relatorio = "relatorio_cnncheckin_$timestamp.txt"
    
    try
        open(arquivo_relatorio, "w") do file
            write(file, "CNN CHECK-IN - RELATÓRIO DO SISTEMA\n")
            write(file, "="^50 * "\n")
            write(file, "Data/Hora: $(Dates.format(now(), "dd/mm/yyyy HH:MM:SS"))\n")
            write(file, "Sistema: $(Sys.KERNEL) $(Sys.ARCH)\n")
            write(file, "Julia: $(VERSION)\n")
            write(file, "Display: $DISPLAY_TYPE\n\n")
            
            write(file, "PASTAS DE FOTOS:\n")
            pastas = ["fotos_rosto", "fotos_rosto_simples", "fotos_rosto_timer", "fotos_rosto_preview"]
            total_fotos = 0
            total_tamanho = 0.0
            
            for pasta in pastas
                if isdir(pasta)
                    fotos = listar_fotos(pasta)
                    tamanho = sum(filesize(foto) for foto in fotos) / (1024*1024)
                    write(file, "  $pasta: $(length(fotos)) fotos, $(round(tamanho, digits=2)) MB\n")
                    total_fotos += length(fotos)
                    total_tamanho += tamanho
                else
                    write(file, "  $pasta: não existe\n")
                end
            end
            
            write(file, "\nTOTAL: $total_fotos fotos, $(round(total_tamanho, digits=2)) MB\n")
            
            write(file, "\nPACOTES GRÁFICOS:\n")
            for (pkg, disponivel) in graphics_available
                status = disponivel ? "DISPONÍVEL" : "NÃO DISPONÍVEL"
                write(file, "  $pkg: $status\n")
            end
        end
        
        println("✅ Relatório exportado: $arquivo_relatorio")
    catch e
        println("❌ Erro ao exportar relatório: $e")
    end
end

# Instruções de uso e execução
if abspath(PROGRAM_FILE) == @__FILE__
    println("🚀 Iniciando CNN Check-in...")
    main()
end

# === INSTRUÇÕES DE USO ===
"""
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
julia cnncheckin_acount_viewer.jl
```

## Estrutura dos arquivos:
- cnncheckin_core.jl: Funções principais e configuração
- cnncheckin_acount_viewer.jl: Interface principal e execução

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

# julia cnncheckin_acount_viewer.jl