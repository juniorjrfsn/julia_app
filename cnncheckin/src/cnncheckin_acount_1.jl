# projeto: cnncheckin
# file: cnncheckin/src/cnncheckin_acount.jl

using VideoIO
using ImageView
using Images
using FileIO
using Dates

# Função para criar diretório se não existir
function criar_diretorio(caminho)
    if !isdir(caminho)
        mkpath(caminho)
        println("Diretório criado: $caminho")
    end
end

# Função para capturar fotos do rosto
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
        
        println("Webcam iniciada! Preparando para capturar $num_fotos fotos...")
        println("Primeira foto em 5 segundos...")
        
        # Aguardar 5 segundos antes da primeira captura
        sleep(5)
        
        foto_count = 0
        
        while foto_count < num_fotos
            try
                # Capturar frame da webcam
                frame = read(camera)
                
                if frame !== nothing
                    # Converter para formato de imagem
                    img = frame
                    
                    # Gerar nome único para a foto
                    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
                    nome_arquivo = "foto_$(foto_count + 1)_$timestamp.jpg"
                    caminho_completo = joinpath(pasta_fotos, nome_arquivo)
                    
                    # Salvar a imagem
                    save(caminho_completo, img)
                    
                    foto_count += 1
                    
                    println("✅ Foto $foto_count/$num_fotos salva: $nome_arquivo")
                    
                    if foto_count < num_fotos
                        println("Próxima foto em $intervalo segundos... Mude de ângulo!")
                        
                        # Mostrar preview da imagem capturada por alguns segundos
                        try
                            imshow(img)
                            sleep(2)  # Mostrar por 2 segundos
                        catch
                            # Se imshow não funcionar, continuar sem preview
                        end
                        
                        sleep(intervalo - 2)  # Resto do intervalo
                    end
                else
                    println("Erro ao capturar frame da webcam")
                    break
                end
                
            catch e
                println("Erro durante captura: $e")
                break
            end
        end
        
        # Fechar webcam
        close(camera)
        
        if foto_count == num_fotos
            println("\n🎉 Captura concluída com sucesso!")
            println("$foto_count fotos salvas na pasta: $pasta_fotos")
        else
            println("\n⚠️ Captura interrompida. $foto_count fotos salvas.")
        end
        
    catch e
        println("Erro ao acessar webcam: $e")
        println("\nDicas para resolver:")
        println("- Verifique se a webcam está conectada")
        println("- Feche outros programas que possam estar usando a webcam")
        println("- Execute o script com permissões adequadas")
    end
end

# Função alternativa com interface mais simples (sem detecção de rosto)
function capturar_fotos_simples()
    pasta_fotos = "fotos_rosto_simples"
    criar_diretorio(pasta_fotos)
    
    println("=== MODO SIMPLES ===")
    println("Pressione ENTER para cada captura, ou digite 'sair' para terminar")
    
    try
        camera = VideoIO.opencamera()
        foto_count = 0
        
        while true
            println("\nPosicione-se e pressione ENTER para capturar (ou 'sair'):")
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
                else
                    println("Erro ao capturar frame")
                end
            catch e
                println("Erro na captura: $e")
            end
        end
        
        close(camera)
        println("\n🎉 $foto_count fotos salvas na pasta: $pasta_fotos")
        
    catch e
        println("Erro ao acessar webcam: $e")
    end
end

# Função principal
function main()
    println("Escolha o modo de captura:")
    println("1 - Automático (captura 10 fotos com intervalo)")
    println("2 - Manual (pressione ENTER para cada foto)")
    print("Escolha (1 ou 2): ")
    
    escolha = readline()
    
    if escolha == "1"
        capturar_fotos_rosto()
    elseif escolha == "2"
        capturar_fotos_simples()
    else
        println("Escolha inválida!")
        main()
    end
end

# Executar o programa
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

# julia cnncheckin_acount.jl

 