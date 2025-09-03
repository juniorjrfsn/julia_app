# projeto: webcamcnn
# file: webcamcnn/src/main_toml_system.jl



# projeto: webcamcnn
# file: webcamcnn/src/main_toml_system.jl

using Dates
using VideoIO
using ImageView
using Images
using FileIO

include("core.jl")
include("weights_manager.jl")
include("weights_utils.jl")
include("pretrain_modified.jl")

using .CNNCheckinCore

# Constantes do sistema
const SYSTEM_VERSION = "2.0"
const WEIGHTS_FILE = "model_weights.toml"

function print_header()
    println("🤖 CNN CHECKIN SYSTEM v$SYSTEM_VERSION")
    println("📊 Sistema com Suporte TOML para Acúmulo de Treinamentos")
    println("=" ^ 60)
    println("📅 $(Dates.format(now(), "dd/mm/yyyy HH:MM:SS"))")
    println()
end

function print_system_status()
    println("📋 STATUS DO SISTEMA:")
    println("-" ^ 30)
    
    # Status dos diretórios
    train_dir_exists = isdir(CNNCheckinCore.TRAIN_DATA_PATH)
    println("📁 Diretório de treino: $(train_dir_exists ? "✅ Existe" : "❌ Ausente") ($(CNNCheckinCore.TRAIN_DATA_PATH))")
    
    # Status dos arquivos de configuração
    config_exists = isfile(CNNCheckinCore.CONFIG_PATH)
    model_exists = isfile(CNNCheckinCore.MODEL_PATH)
    weights_toml_exists = isfile(WEIGHTS_FILE)
    
    println("⚙️  Arquivo de config: $(config_exists ? "✅ Existe" : "❌ Ausente") ($(CNNCheckinCore.CONFIG_PATH))")
    println("🧠 Modelo JLD2: $(model_exists ? "✅ Existe" : "❌ Ausente") ($(CNNCheckinCore.MODEL_PATH))")
    println("📊 Pesos TOML: $(weights_toml_exists ? "✅ Existe" : "❌ Ausente") ($WEIGHTS_FILE)")
    
    # Status dos dados de treino
    if train_dir_exists
        dados_ok, msg_dados = verificar_dados_treino()
        println("📸 Dados de treino: $(dados_ok ? "✅" : "⚠️ ") $msg_dados")
    end
    
    # Status dos treinamentos salvos
    if weights_toml_exists
        try
            data = TOML.parsefile(WEIGHTS_FILE)
            num_trainings = length(get(data, "trainings", Dict()))
            if num_trainings > 0
                latest = get(get(data, "summary", Dict()), "latest_training", "N/A")
                println("🏆 Treinamentos salvos: ✅ $num_trainings (último: $latest)")
            else
                println("🏆 Treinamentos salvos: ⚠️  Arquivo existe mas vazio")
            end
        catch
            println("🏆 Treinamentos salvos: ❌ Erro ao ler arquivo")
        end
    end
    
    println()
end

# Verificar se há dados suficientes para treino
function verificar_dados_treino()
    if !isdir(CNNCheckinCore.TRAIN_DATA_PATH)
        return false, "Diretório de dados não existe"
    end
    
    arquivos = readdir(CNNCheckinCore.TRAIN_DATA_PATH)
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    
    imagens_validas = 0
    pessoas = Set{String}()
    
    for arquivo in arquivos
        ext = lowercase(splitext(arquivo)[2])
        if ext in image_extensions
            # Verificar se a imagem é válida
            caminho = joinpath(CNNCheckinCore.TRAIN_DATA_PATH, arquivo)
            if CNNCheckinCore.validate_image_file(caminho)
                imagens_validas += 1
                pessoa = CNNCheckinCore.extract_person_name(arquivo)
                push!(pessoas, pessoa)
            end
        end
    end
    
    num_pessoas = length(pessoas)
    
    if num_pessoas < 1
        return false, "Nenhuma pessoa encontrada nos dados"
    end
    
    if imagens_validas < 5
        return false, "Poucas imagens válidas encontradas ($imagens_validas). Mínimo recomendado: 5"
    end
    
    return true, "Dados válidos: $num_pessoas pessoa(s), $imagens_validas imagem(s)"
end

function show_training_summary()
    if !isfile(WEIGHTS_FILE)
        println("📭 Nenhum treinamento encontrado ainda")
        return
    end
    
    println("📚 RESUMO DOS TREINAMENTOS SALVOS:")
    println("-" ^ 40)
    
    try
        data = TOML.parsefile(WEIGHTS_FILE)
        trainings = get(data, "trainings", Dict())
        
        if isempty(trainings)
            println("📭 Nenhum treinamento salvo ainda")
            return
        end
        
        # Ordenar por timestamp
        training_list = collect(trainings)
        sort!(training_list, by=x -> x[2]["metadata"]["timestamp"])
        
        println("📢 Total de treinamentos: $(length(trainings))")
        println("📈 Últimos 3 treinamentos:")
        
        recent_trainings = training_list[max(1, end-2):end]
        for (training_id, training_data) in recent_trainings
            meta = training_data["metadata"]
            acc = round(meta["final_accuracy"] * 100, digits=1)
            persons = join(meta["person_names"], ", ")
            date = split(meta["timestamp"], "T")[1]
            
            println("   🏆 $training_id")
            println("      📅 $date | 🎯 $acc% | 👥 $(length(meta["person_names"])) pessoas")
            println("      👤 $persons")
        end
        
        if haskey(data, "summary")
            summary = data["summary"]
            all_persons = get(summary, "all_persons", [])
            println("\n👥 Total de pessoas no sistema: $(length(all_persons))")
            if !isempty(all_persons)
                println("📝 Pessoas: $(join(all_persons, ", "))")
            end
        end
        
    catch e
        println("❌ Erro ao ler resumo: $e")
    end
    
    println()
end

# Função para capturar fotos da webcam (integrada no sistema principal)
function capturar_fotos_rosto()
    println("=== SISTEMA DE CAPTURA E TREINAMENTO FACIAL ===")
    println()
    
    # Solicitar nome da pessoa
    print("Digite o nome da pessoa (será usado para treinar o modelo): ")
    nome_pessoa = strip(readline())
    
    if isempty(nome_pessoa)
        println("❌ Nome não pode estar vazio!")
        return false
    end
    
    # Configurações
    pasta_fotos = CNNCheckinCore.TRAIN_DATA_PATH
    num_fotos = 10  # Número de fotos a capturar
    intervalo = 3   # Intervalo em segundos entre capturas
    
    # Criar diretório para as fotos
    CNNCheckinCore.criar_diretorio(pasta_fotos)
    
    println()
    println("=== CAPTURADOR DE FOTOS FACIAIS ===")
    println("Instruções:")
    println("- Posicione-se em frente à webcam")
    println("- Mude de ângulo a cada captura (frontal, perfil esquerdo, perfil direito, etc.)")
    println("- Mantenha boa iluminação")
    println("- Pressione ENTER para iniciar")
    println("- Pressione 'q' na janela da webcam para sair antecipadamente")
    println()
    println("Pessoa: $nome_pessoa")
    println("Fotos a capturar: $num_fotos")
    println()
    
    readline()  # Aguarda pressionar ENTER
    
    try
        # Abrir webcam (geralmente índice 0 para webcam padrão)
        camera = VideoIO.opencamera()
        
        println("📹 Webcam iniciada! Preparando para capturar $num_fotos fotos...")
        println("Primeira foto em 5 segundos...")
        
        # Aguardar 5 segundos antes da primeira captura
        sleep(5)
        
        foto_count = 0
        fotos_salvas = String[]  # Lista das fotos salvas
        
        while foto_count < num_fotos
            try
                # Capturar frame da webcam
                frame = read(camera)
                
                if frame !== nothing
                    # Converter para formato de imagem
                    img = frame
                    
                    # Gerar nome único para a foto usando o nome da pessoa
                    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
                    nome_arquivo = "$(nome_pessoa)_$(foto_count + 1)_$timestamp.jpg"
                    caminho_completo = joinpath(pasta_fotos, nome_arquivo)
                    
                    # Salvar a imagem
                    save(caminho_completo, img)
                    
                    foto_count += 1
                    push!(fotos_salvas, caminho_completo)
                    
                    println("✅ Foto $foto_count/$num_fotos salva: $nome_arquivo")
                    
                    if foto_count < num_fotos
                        println("Próxima foto em $intervalo segundos... Mude de ângulo!")
                        
                        # Mostrar preview da imagem capturada por alguns segundos
                        try
                            imshow(img)
                            sleep(2)  # Mostrar por 2 segundos
                        catch
                            # Se imshow não funcionar, continuar sem preview
                            println("   (Preview não disponível)")
                        end
                        
                        sleep(intervalo - 2)  # Resto do intervalo
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
            println("$foto_count fotos salvas na pasta: $pasta_fotos")
            println("\nFotos capturadas:")
            for (i, foto) in enumerate(fotos_salvas)
                println("   $i. $(basename(foto))")
            end
            return true
        else
            println("\n⚠️ Captura interrompida. $foto_count fotos salvas.")
            return foto_count > 0  # Retorna true se pelo menos uma foto foi salva
        end
        
    catch e
        println("❌ Erro ao acessar webcam: $e")
        println("\nDicas para resolver:")
        println("- Verifique se a webcam está conectada")
        println("- Feche outros programas que possam estar usando a webcam")
        println("- Execute o script com permissões adequadas")
        println("- Certifique-se de que os drivers da webcam estão instalados")
        return false
    end
end

# Função alternativa com interface mais simples (sem detecção de rosto)
function capturar_fotos_simples()
    println("=== MODO SIMPLES ===")
    
    # Solicitar nome da pessoa
    print("Digite o nome da pessoa: ")
    nome_pessoa = strip(readline())
    
    if isempty(nome_pessoa)
        println("❌ Nome não pode estar vazio!")
        return false
    end
    
    pasta_fotos = CNNCheckinCore.TRAIN_DATA_PATH
    CNNCheckinCore.criar_diretorio(pasta_fotos)
    
    println("Pressione ENTER para cada captura, ou digite 'sair' para terminar")
    println("Pessoa: $nome_pessoa")
    println()
    
    try
        camera = VideoIO.opencamera()
        foto_count = 0
        fotos_salvas = String[]
        
        while true
            println("Posicione-se e pressione ENTER para capturar (ou 'sair'):")
            entrada = readline()
            
            if lowercase(strip(entrada)) == "sair"
                break
            end
            
            try
                frame = read(camera)
                if frame !== nothing
                    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
                    nome_arquivo = "$(nome_pessoa)_$(foto_count + 1)_$timestamp.jpg"
                    caminho_completo = joinpath(pasta_fotos, nome_arquivo)
                    
                    save(caminho_completo, frame)
                    foto_count += 1
                    push!(fotos_salvas, caminho_completo)
                    
                    println("✅ Foto $foto_count salva: $nome_arquivo")
                    
                    # Mostrar preview se possível
                    try
                        imshow(frame)
                        sleep(1)
                    catch
                        println("   (Preview não disponível)")
                    end
                else
                    println("❌ Erro ao capturar frame")
                end
            catch e
                println("❌ Erro na captura: $e")
            end
        end
        
        close(camera)
        
        if foto_count > 0
            println("\n🎉 $foto_count fotos salvas na pasta: $pasta_fotos")
            println("\nFotos capturadas:")
            for (i, foto) in enumerate(fotos_salvas)
                println("   $i. $(basename(foto))")
            end
            return true
        else
            println("Nenhuma foto foi capturada.")
            return false
        end
        
    catch e
        println("❌ Erro ao acessar webcam: $e")
        return false
    end
end

function capture_and_train_workflow()
    println("\n🎥 CAPTURA E TREINAMENTO INTEGRADOS")
    println("=" ^ 50)
    
    # Verificar se já existem dados de treino
    dados_ok, msg_dados = verificar_dados_treino()
    
    if dados_ok
        println("📊 Status dos dados existentes: $msg_dados")
        println()
        println("Escolha uma opção:")
        println("1 - Capturar mais fotos (modo automático)")
        println("2 - Capturar mais fotos (modo manual)")
        println("3 - Iniciar treinamento com dados existentes")
        println("4 - Voltar")
    else
        println("📊 Status dos dados: $msg_dados")
        println()
        println("Escolha o modo de captura:")
        println("1 - Automático (captura 10 fotos com intervalo)")
        println("2 - Manual (pressione ENTER para cada foto)")
        println("3 - Voltar")
    end
    
    print("Escolha: ")
    escolha = readline()
    
    captura_realizada = false
    
    if escolha == "1"
        println("\n🎥 Iniciando captura automática...")
        captura_realizada = capturar_fotos_rosto()
    elseif escolha == "2"
        println("\n🎥 Iniciando captura manual...")
        captura_realizada = capturar_fotos_simples()
    elseif escolha == "3"
        if dados_ok
            println("\n🧠 Iniciando treinamento...")
            return iniciar_treinamento()
        else
            println("❌ Não é possível treinar sem dados válidos!")
            return false
        end
    elseif escolha == "4" || escolha == "3"
        println("👋 Voltando...")
        return false
    else
        println("❌ Escolha inválida!")
        return capture_and_train_workflow()  # Recursivamente chama o menu novamente
    end
    
    # Se chegou até aqui e houve captura, perguntar sobre treino
    if captura_realizada
        println()
        print("Deseja iniciar o treinamento agora? (s/n): ")
        resposta = strip(lowercase(readline()))
        
        if resposta == "s" || resposta == "sim" || resposta == "y" || resposta == "yes"
            println("\n🧠 Iniciando treinamento...")
            return iniciar_treinamento()
        else
            println("Treinamento pode ser executado posteriormente.")
            println("Para treinar, execute novamente o sistema principal.")
            return true
        end
    end
    
    return captura_realizada
end

# Função para iniciar o treinamento
function iniciar_treinamento()
    println("🚀 INICIANDO FASE DE TREINAMENTO")
    println("=" ^ 40)
    
    # Verificar dados antes de treinar
    dados_ok, msg_dados = verificar_dados_treino()
    if !dados_ok
        println("❌ $msg_dados")
        return false
    end
    
    println("✅ $msg_dados")
    println()
    
    try
        # Executar o pré-treinamento
        success = pretrain_command()
        
        if success
            println("\n🎉 SISTEMA TREINADO COM SUCESSO!")
            println("=" ^ 40)
            println("O modelo está pronto para identificação facial.")
            println()
            println("Próximos passos:")
            println("- Use o modelo treinado para identificação")
            println("- Adicione mais pessoas se necessário")
            println("- Execute treino incremental para melhorar a performance")
        else
            println("\n❌ FALHA NO TREINAMENTO")
            println("Verifique os dados e tente novamente.")
        end
        
        return success
        
    catch e
        println("❌ Erro durante o treinamento: $e")
        return false
    end
end

function main_menu()
    while true
        print_header()
        print_system_status()
        show_training_summary()
        
        println("🎯 MENU PRINCIPAL:")
        println("-" ^ 20)
        println("1️⃣  - Capturar fotos e treinar novo modelo")
        println("2️⃣  - Gerenciar treinamentos (listar, comparar, carregar)")
        println("3️⃣  - Utilitários de pesos TOML (backup, análise, limpeza)")
        println("4️⃣  - Executar apenas novo treinamento (dados existentes)")
        println("5️⃣  - Mostrar detalhes do sistema")
        println("6️⃣  - Sair")
        println()
        
        print("🎮 Escolha uma opção: ")
        choice = strip(readline())
        
        if choice == "1"
            resultado = capture_and_train_workflow()
            if resultado
                println("\n✅ Captura e treinamento concluídos!")
            end
            println("Press Enter para continuar...")
            readline()
            
        elseif choice == "2"
            println("\n🎛️  MODO: GERENCIAMENTO DE TREINAMENTOS")
            println("=" ^ 40)
            training_management_menu()
            
        elseif choice == "3"
            println("\n🛠️  MODO: UTILITÁRIOS DE PESOS")
            println("=" ^ 40)
            weights_utilities_menu()
            
        elseif choice == "4"
            println("\n🧠 MODO: APENAS TREINAMENTO")
            println("=" ^ 40)
            
            dados_ok, msg = verificar_dados_treino()
            if !dados_ok
                println("❌ $msg")
                println("Use a opção 1 para capturar fotos primeiro")
                println("Press Enter para continuar...")
                readline()
                continue
            end
            
            println("✅ $msg")
            print("Deseja continuar com o treinamento? (s/n): ")
            confirm = strip(lowercase(readline()))
            
            if confirm in ["s", "sim", "y", "yes"]
                success = pretrain_command()
                if success
                    println("\n🎉 Treinamento concluído com sucesso!")
                else
                    println("\n❌ Treinamento falhou")
                end
            end
            println("Press Enter para continuar...")
            readline()
            
        elseif choice == "5"
            show_system_details()
            
        elseif choice == "6"
            println("\n👋 Encerrando sistema...")
            println("📊 Obrigado por usar o CNN Checkin System v$SYSTEM_VERSION!")
            break
            
        else
            println("\nArmações Oçulos OpçãoLorenzo invália! Tente novamente.")
            println("Press Enter para continuar...")
            readline()
        end
    end
end

function show_system_details()
    println("\n📋 DETALHES DO SISTEMA:")
    println("=" ^ 50)
    
    println("🔧 Configurações:")
    println("   - Versão: $SYSTEM_VERSION")
    println("   - Tamanho da imagem: $(CNNCheckinCore.IMG_SIZE)")
    println("   - Batch size: $(CNNCheckinCore.BATCH_SIZE)")
    println("   - Épocas pré-treino: $(CNNCheckinCore.PRETRAIN_EPOCHS)")
    println("   - Learning rate: $(CNNCheckinCore.LEARNING_RATE)")
    
    println("\n📁 Caminhos de arquivos:")
    println("   - Dados de treino: $(CNNCheckinCore.TRAIN_DATA_PATH)")
    println("   - Modelo JLD2: $(CNNCheckinCore.MODEL_PATH)")
    println("   - Configuração: $(CNNCheckinCore.CONFIG_PATH)")
    println("   - Pesos TOML: $WEIGHTS_FILE")
    println("   - Dados modelo TOML: $(CNNCheckinCore.MODEL_DATA_TOML_PATH)")
    
    println("\n💾 Arquivos existentes:")
    files_info = [
        (CNNCheckinCore.CONFIG_PATH, "Config TOML"),
        (CNNCheckinCore.MODEL_PATH, "Modelo JLD2"),
        (CNNCheckinCore.MODEL_DATA_TOML_PATH, "Dados Modelo TOML"),
        (WEIGHTS_FILE, "Pesos TOML")
    ]
    
    for (filepath, description) in files_info
        exists = isfile(filepath)
        size_info = ""
        if exists
            try
                size_kb = round(filesize(filepath) / 1024, digits=1)
                size_info = " ($(size_kb) KB)"
            catch
                size_info = " (tamanho desconhecido)"
            end
        end
        status = exists ? "✅ Existe$size_info" : "❌ Ausente"
        println("   - $description: $status")
    end
    
    # Informações do arquivo de pesos se existir
    if isfile(WEIGHTS_FILE)
        println("\n📊 Análise do arquivo de pesos:")
        try
            data = TOML.parsefile(WEIGHTS_FILE)
            trainings = get(data, "trainings", Dict())
            
            if !isempty(trainings)
                total_size = 0
                for (_, training_data) in trainings
                    if haskey(training_data, "model_stats")
                        total_size += get(training_data["model_stats"], "total_parameters", 0)
                    end
                end
                
                avg_params = total_size > 0 ? div(total_size, length(trainings)) : 0
                
                println("   - Total de treinamentos: $(length(trainings))")
                println("   - Média de parâmetros por modelo: $avg_params")
                println("   - Tamanho estimado por modelo: $(round(avg_params * 4 / (1024^2), digits=1)) MB")
            end
        catch e
            println("   - Erro ao analisar: $e")
        end
    end
    
    println("\n🚀 Funcionalidades disponíveis:")
    println("   ✅ Captura de fotos da webcam")
    println("   ✅ Treinamento de modelos CNN")
    println("   ✅ Salvamento de pesos em TOML")
    println("   ✅ Acúmulo de múltiplos treinamentos")
    println("   ✅ Comparação entre treinamentos")
    println("   ✅ Análise de evolução")
    println("   ✅ Backup e exportação")
    println("   ✅ Validação de integridade")
    
    println("\n📖 Para mais informações:")
    println("   - Veja os arquivos de código fonte")
    println("   - Consulte a documentação do projeto")
    
    println("\nPress Enter para continuar...")
    readline()
end

# Função de inicialização
function initialize_system()
    println("🚀 Inicializando CNN Checkin System v$SYSTEM_VERSION...")
    
    # Criar diretórios necessários
    CNNCheckinCore.criar_diretorio(CNNCheckinCore.TRAIN_DATA_PATH)
    
    # Verificar dependências básicas
    required_packages = ["Flux", "Images", "TOML", "JLD2", "VideoIO"]
    missing_packages = []
    
    for pkg in required_packages
        try
            eval(Meta.parse("using $pkg"))
        catch
            push!(missing_packages, pkg)
        end
    end
    
    if !isempty(missing_packages)
        println("⚠️  Pacotes faltando: $(join(missing_packages, ", "))")
        println("Execute: using Pkg; Pkg.add([\"$(join(missing_packages, "\", \""))\"])")
        return false
    end
    
    println("✅ Sistema inicializado com sucesso!")
    return true
end

# Função principal
function main()
    if !initialize_system()
        println("❌ Falha na inicialização do sistema")
        return false
    end
    
    try
        main_menu()
        return true
    catch e
        println("❌ Erro durante execução: $e")
        return false
    end
end

# Executar se chamado diretamente
if abspath(PROGRAM_FILE) == @__FILE__
    println("🎬 Executando CNN Checkin System v$SYSTEM_VERSION")
    success = main()
    println(success ? "✅ Sistema encerrado normalmente" : "❌ Sistema encerrado com erros")
end

# Export functions
export main, initialize_system, show_system_details, verificar_dados_treino, 
       capture_and_train_workflow, capturar_fotos_rosto, capturar_fotos_simples

# Executar sistema completo
# julia main_toml_system.jl

# Opções disponíveis:
# 1 - Capturar fotos + treinar (gera TOML automaticamente)
# 2 - Gerenciar treinamentos existentes
# 3 - Utilitários de análise e manutenção
# 4 - Apenas treinar com dados existentes


# Como usar os scripts corrigidos:
# Opção 1: Sistema Principal Integrado
# julia main_toml_system.jl

# Opção 2: Script de Inicialização (Recomendado)
# julia start_system.jl

# Opção 3: Apenas Captura (Independente)
# julia capture_and_train.jl