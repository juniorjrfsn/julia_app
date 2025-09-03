# projeto: webcamcnn
# file: webcamcnn/src/capture_and_train.jl

# projeto: webcamcnn
# file: webcamcnn/src/capture_and_train.jl

using VideoIO
using ImageView
using Images
using FileIO
using Dates
using TOML

include("core.jl")
include("pretrain_modified.jl")

using .CNNCheckinCore

# Função principal para capturar fotos da webcam
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

# Função para verificar se há dados suficientes para treino
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

# Função principal que combina captura e treino
function main()
    println("🤖 SISTEMA CNN CHECKIN - CAPTURA E TREINAMENTO")
    println("=" ^ 50)
    println()
    
    # Verificar se já existem dados de treino
    dados_ok, msg_dados = verificar_dados_treino()
    
    if dados_ok
        println("📊 Status dos dados existentes: $msg_dados")
        println()
        println("Escolha uma opção:")
        println("1 - Capturar mais fotos (modo automático)")
        println("2 - Capturar mais fotos (modo manual)")
        println("3 - Iniciar treinamento com dados existentes")
        println("4 - Sair")
    else
        println("📊 Status dos dados: $msg_dados")
        println()
        println("Escolha o modo de captura:")
        println("1 - Automático (captura 10 fotos com intervalo)")
        println("2 - Manual (pressione ENTER para cada foto)")
        println("3 - Sair")
    end
    
    print("Escolha: ")
    escolha = readline()
    
    captura_realizada = false
    
    if escolha == "1"
        println("\n📥 Iniciando captura automática...")
        captura_realizada = capturar_fotos_rosto()
    elseif escolha == "2"
        println("\n📥 Iniciando captura manual...")
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
        println("👋 Saindo...")
        return false
    else
        println("❌ Escolha inválida!")
        return main()  # Recursivamente chama o menu novamente
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
            println("Para treinar, execute: julia capture_and_train.jl")
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
        # Incluir e executar o pré-treinamento
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

# Executar o programa se chamado diretamente
if abspath(PROGRAM_FILE) == @__FILE__
    resultado = main()
    if resultado
        println("\n✅ Programa executado com sucesso!")
    else
        println("\n⚠️ Programa finalizado.")
    end
end

# Export functions
export capturar_fotos_rosto, capturar_fotos_simples, verificar_dados_treino, 
       main, iniciar_treinamento