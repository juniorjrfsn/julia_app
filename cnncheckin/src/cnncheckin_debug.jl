using Flux
using JLD2
using Statistics
using Images
using FileIO

include("cnncheckin_core.jl")
using .CNNCheckinCore

# Função para debug detalhado do modelo e imagem
function debug_model_and_image(image_path::String)
    println("🔍 Debug Detalhado - Sistema de Reconhecimento Facial")
    println("="^60)
    
    # 1. Verificar arquivos
    println("\n📁 Verificação de Arquivos:")
    files_to_check = [
        ("Imagem", image_path),
        ("Modelo", CNNCheckinCore.MODEL_PATH),
        ("Configuração", CNNCheckinCore.CONFIG_PATH),
        ("Dados TOML", CNNCheckinCore.MODEL_DATA_TOML_PATH)
    ]
    
    for (name, path) in files_to_check
        status = isfile(path) ? "✅" : "❌"
        println("   $status $name: $path")
    end
    
    # 2. Testar carregamento da imagem
    println("\n🖼️ Debug da Imagem:")
    try
        println("   📂 Carregando imagem...")
        img = load(image_path)
        println("   ✅ Imagem carregada com sucesso")
        println("   📏 Tipo original: $(typeof(img))")
        println("   📏 Dimensões originais: $(size(img))")
        
        # Testar preprocessamento
        println("   🔄 Testando preprocessamento...")
        img_arrays = CNNCheckinCore.preprocess_image(image_path; augment=false)
        
        if img_arrays === nothing
            println("   ❌ Falha no preprocessamento")
            return false
        else
            println("   ✅ Preprocessamento bem-sucedido")
            img_array = img_arrays[1]
            println("   📏 Dimensões após processamento: $(size(img_array))")
            println("   📊 Tipo dos dados: $(eltype(img_array))")
            println("   📊 Estatísticas dos pixels:")
            println("      - Mínimo: $(round(minimum(img_array), digits=4))")
            println("      - Máximo: $(round(maximum(img_array), digits=4))")
            println("      - Média: $(round(mean(img_array), digits=4))")
            println("      - Desvio padrão: $(round(std(img_array), digits=4))")
        end
    catch e
        println("   ❌ Erro ao carregar/processar imagem: $e")
        return false
    end
    
    # 3. Debug do modelo
    println("\n🧠 Debug do Modelo:")
    try
        # Carregar configuração
        if !isfile(CNNCheckinCore.CONFIG_PATH)
            println("   ❌ Arquivo de configuração não encontrado")
            return false
        end
        
        config = CNNCheckinCore.load_config(CNNCheckinCore.CONFIG_PATH)
        println("   ✅ Configuração carregada")
        println("   📊 Número de classes: $(config["model"]["num_classes"])")
        println("   👥 Pessoas: $(join(config["data"]["person_names"], ", "))")
        
        # Carregar modelo
        if !isfile(CNNCheckinCore.MODEL_PATH)
            println("   ❌ Arquivo do modelo não encontrado")
            return false
        end
        
        data = load(CNNCheckinCore.MODEL_PATH)
        model = data["model_data"]["model_state"]
        println("   ✅ Modelo carregado")
        println("   🏗️ Arquitetura do modelo:")
        
        # Mostrar cada camada
        for (i, layer) in enumerate(model)
            println("      $i. $(typeof(layer))")
        end
        
        # 4. Teste de predição passo a passo
        println("\n🔬 Teste de Predição Passo a Passo:")
        
        # Preparar entrada
        img_arrays = CNNCheckinCore.preprocess_image(image_path; augment=false)
        img_array = img_arrays[1]
        img_tensor = reshape(img_array, size(img_array)..., 1)
        
        println("   📥 Tensor de entrada preparado: $(size(img_tensor))")
        
        # Executar cada camada individualmente
        current_output = img_tensor
        println("   🔄 Executando cada camada...")
        
        for (i, layer) in enumerate(model)
            try
                println("      Camada $i ($(typeof(layer))): entrada $(size(current_output))")
                current_output = layer(current_output)
                println("         → saída $(size(current_output))")
                
                # Mostrar estatísticas da saída
                if length(current_output) < 100  # Só para saídas pequenas
                    println("         📊 Min: $(round(minimum(current_output), digits=4)), Max: $(round(maximum(current_output), digits=4))")
                end
            catch e
                println("      ❌ Erro na camada $i: $e")
                println("         Tipo do erro: $(typeof(e))")
                return false
            end
        end
        
        # 5. Análise da saída final
        println("\n📊 Análise da Saída Final:")
        logits = current_output
        println("   📏 Dimensões dos logits: $(size(logits))")
        println("   🔢 Logits brutos: $(vec(logits))")
        
        # Aplicar softmax
        try
            logits_vec = Float32.(vec(logits))
            max_logit = maximum(logits_vec)
            exp_logits = exp.(logits_vec .- max_logit)
            sum_exp = sum(exp_logits)
            probabilities = exp_logits ./ sum_exp
            
            println("   📊 Probabilidades:")
            person_names = config["data"]["person_names"]
            for (i, (prob, name)) in enumerate(zip(probabilities, person_names))
                println("      $i. $name: $(round(prob*100, digits=2))%")
            end
            
            pred_class = argmax(probabilities)
            confidence = probabilities[pred_class]
            predicted_person = person_names[pred_class]
            
            println("\n🎯 Resultado Final:")
            println("   👤 Pessoa predita: $predicted_person")
            println("   📈 Confiança: $(round(confidence*100, digits=2))%")
            
        catch e
            println("   ❌ Erro no cálculo de softmax: $e")
            return false
        end
        
        return true
        
    catch e
        println("   ❌ Erro geral no debug: $e")
        println("   🔍 Tipo: $(typeof(e))")
        return false
    end
end

# Função para testar múltiplas imagens
function test_multiple_images(image_dir::String)
    println("🧪 Teste em Múltiplas Imagens")
    println("📁 Diretório: $image_dir")
    
    if !isdir(image_dir)
        println("❌ Diretório não encontrado: $image_dir")
        return false
    end
    
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    image_files = []
    
    for filename in readdir(image_dir)
        ext = lowercase(splitext(filename)[2])
        if ext in image_extensions
            push!(image_files, joinpath(image_dir, filename))
        end
    end
    
    if isempty(image_files)
        println("❌ Nenhuma imagem encontrada no diretório")
        return false
    end
    
    println("📊 Encontradas $(length(image_files)) imagens para teste")
    
    success_count = 0
    for (i, img_path) in enumerate(image_files)
        println("\n" * "="^40)
        println("🖼️ Teste $i/$(length(image_files)): $(basename(img_path))")
        if debug_model_and_image(img_path)
            success_count += 1
            println("✅ Sucesso")
        else
            println("❌ Falha")
        end
    end
    
    println("\n📊 Resumo do Teste:")
    println("   ✅ Sucessos: $success_count/$(length(image_files))")
    println("   ❌ Falhas: $(length(image_files) - success_count)/$(length(image_files))")
    
    return success_count > 0
end

# Executar comando se for chamado diretamente
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        println("❌ Erro: Parâmetro necessário")
        println("Uso:")
        println("  julia cnncheckin_debug.jl <caminho_da_imagem>        # Debug de uma imagem")
        println("  julia cnncheckin_debug.jl --dir <diretório>         # Teste múltiplas imagens")
        println()
        println("Exemplos:")
        println("  julia cnncheckin_debug.jl ../../../dados/fotos_teste/teste.png")
        println("  julia cnncheckin_debug.jl --dir ../../../dados/fotos_teste/")
    else
        if ARGS[1] == "--dir" && length(ARGS) >= 2
            success = test_multiple_images(ARGS[2])
        else
            success = debug_model_and_image(ARGS[1])
        end
        
        if success
            println("\n✅ Debug concluído com sucesso!")
        else
            println("\n💥 Falhas encontradas no debug")
        end
    end
end