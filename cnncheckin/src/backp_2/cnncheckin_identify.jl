using Flux
using JLD2
using Statistics

include("cnncheckin_core.jl")
using .CNNCheckinCore

# Função para carregar modelo, configuração e dados do modelo (CORRIGIDA)
function load_model_and_config(model_filepath::String, config_filepath::String)
    println("📂 Carregando modelo, configuração e dados do modelo...")
    
    config = CNNCheckinCore.load_config(config_filepath)
    CNNCheckinCore.validate_config(config)
    
    if !isfile(model_filepath)
        error("Arquivo do modelo não encontrado: $model_filepath")
    end
    
    try
        data = load(model_filepath)
        model_data = data["model_data"]
        model_state = model_data["model_state"]
        
        # CORREÇÃO: Usar person_names da configuração (ordem correta)
        person_names = config["data"]["person_names"]
        num_classes = config["model"]["num_classes"]
        
        # Verificar se há person_names salvos no modelo também
        if haskey(model_data, "person_names")
            saved_names = model_data["person_names"]
            if saved_names != person_names
                println("⚠️ Aviso: Nomes no modelo diferem da configuração")
                println("   Modelo: $saved_names")
                println("   Config: $person_names")
                println("   Usando da configuração...")
            end
        end
        
        model_data_toml = CNNCheckinCore.load_model_data_toml(CNNCheckinCore.MODEL_DATA_TOML_PATH)
        
        println("✅ Modelo e configuração carregados com sucesso!")
        println("📊 Informações do modelo:")
        println("   - Classes: $num_classes")
        println("   - Pessoas: $(join(person_names, ", "))")
        println("   - Acurácia: $(round(config["training"]["final_accuracy"]*100, digits=2))%")
        println("   - Criado em: $(config["data"]["timestamp"])")
        
        # Verificar mapeamento
        println("🏷️ Mapeamento correto:")
        for (i, name) in enumerate(person_names)
            println("   - Índice $i: $name")
        end
        
        if model_data_toml !== nothing
            println("   - Dados TOML disponíveis: ✅")
            total_params = get(get(model_data_toml, "weights_summary", Dict()), "total_parameters", 0)
            if total_params > 0
                println("   - Total de parâmetros: $(total_params)")
            end
        else
            println("   - Dados TOML disponíveis: ❌")
        end
        
        return model_state, person_names, config, model_data_toml
    catch e
        error("Erro ao carregar modelo: $e")
    end
end

# Função para fazer predição e registrar exemplo (VERSÃO TOTALMENTE CORRIGIDA)
function predict_person(model, person_names, img_path::String; save_example::Bool = true)
    println("🔄 Processando imagem: $img_path")
    
    # Preprocessar imagem
    img_arrays = CNNCheckinCore.preprocess_image(img_path; augment=false)
    
    if img_arrays === nothing || length(img_arrays) == 0
        println("❌ Não foi possível processar a imagem")
        return nothing, 0.0
    end
    
    img_array = img_arrays[1]
    println("📏 Dimensões da imagem processada: $(size(img_array))")
    
    # Preparar tensor de entrada - garantindo formato correto
    img_tensor = reshape(img_array, size(img_array)..., 1)
    println("📏 Dimensões do tensor de entrada: $(size(img_tensor))")
    
    try
        println("🧠 Executando predição...")
        
        # Executar o modelo
        logits = model(img_tensor)
        println("📏 Dimensões da saída do modelo: $(size(logits))")
        println("🔢 Logits brutos: $(vec(logits))")
        
        # Verificar compatibilidade de dimensões
        if size(logits, 1) != length(person_names)
            error("Dimensão de saída do modelo ($(size(logits, 1))) não corresponde ao número de classes ($(length(person_names)))")
        end
        
        # Converter para Float32 e aplicar softmax de forma robusta
        logits_vec = Float32.(vec(logits))
        println("🔢 Logits como vetor Float32: $logits_vec")
        
        # Aplicar softmax manualmente para maior controle
        max_logit = maximum(logits_vec)
        exp_logits = exp.(logits_vec .- max_logit)
        sum_exp = sum(exp_logits)
        probabilities = exp_logits ./ sum_exp
        
        println("📊 Probabilidades: $probabilities")
        
        # Mostrar probabilidade para cada pessoa
        println("📊 Probabilidades por pessoa:")
        for (i, (name, prob)) in enumerate(zip(person_names, probabilities))
            println("   $i. $name: $(round(prob*100, digits=2))%")
        end
        
        # Encontrar a classe com maior probabilidade
        pred_class = argmax(probabilities)
        confidence = probabilities[pred_class]
        
        println("🎯 Classe predita: $pred_class")
        println("📈 Confiança: $(round(confidence*100, digits=2))%")
        
        # CORREÇÃO PRINCIPAL: Verificar se o índice é válido
        if pred_class <= 0 || pred_class > length(person_names)
            println("⚠️ Índice de classe inválido: $pred_class")
            return "Desconhecido", Float64(confidence)
        end
        
        # CORREÇÃO: O argmax já retorna o índice correto para Julia (1-based)
        person_name = person_names[pred_class]
        println("👤 Pessoa identificada: $person_name")
        
        # Salvar exemplo se solicitado
        if save_example
            try
                CNNCheckinCore.add_prediction_example_to_toml(img_path, person_name, Float64(confidence))
                println("💾 Exemplo salvo com sucesso")
            catch e
                println("⚠️ Erro ao salvar exemplo: $e")
            end
        end
        
        return person_name, Float64(confidence)
        
    catch e
        println("❌ Erro ao realizar predição: $e")
        println("📏 Detalhes do erro: $(typeof(e))")
        if isa(e, BoundsError)
            println("📏 Erro de bounds - verificando dimensões...")
            println("   - Tamanho do tensor: $(size(img_tensor))")
            println("   - Número de pessoas: $(length(person_names))")
        end
        return nothing, 0.0
    end
end

# Função de identificação (VERSÃO FINAL CORRIGIDA)
function identify_command(image_path::String)
    println("🔍 Sistema de Reconhecimento Facial - Modo Identificação")
    println("📸 Analisando imagem: $image_path")
    
    try
        # Verificações preliminares
        if !isfile(image_path)
            error("Arquivo de imagem não encontrado: $image_path")
        end
        if !isfile(CNNCheckinCore.MODEL_PATH)
            error("Modelo não encontrado! Execute primeiro: julia cnncheckin_train.jl")
        end
        if !isfile(CNNCheckinCore.CONFIG_PATH)
            error("Configuração não encontrada! Execute primeiro: julia cnncheckin_train.jl")
        end
        
        # Carregar modelo e configuração
        model, person_names, config, model_data_toml = load_model_and_config(CNNCheckinCore.MODEL_PATH, CNNCheckinCore.CONFIG_PATH)
        
        println("🔄 Iniciando processo de predição...")
        
        # Fazer predição
        person_name, confidence = predict_person(model, person_names, image_path; save_example=true)
        
        if person_name === nothing
            println("❌ Não foi possível processar a imagem")
            return false
        end
        
        println("\n" * "="^50)
        println("🎯 Resultado da Identificação:")
        println("   👤 Pessoa: $person_name")
        println("   📈 Confiança: $(round(confidence*100, digits=2))%")
        println("="^50)
        
        println("\n📊 Informações do Modelo:")
        println("   🧠 Acurácia do modelo: $(round(config["training"]["final_accuracy"]*100, digits=2))%")
        println("   📅 Treinado em: $(config["data"]["timestamp"])")
        println("   🎓 Melhor epoch: $(config["training"]["best_epoch"])")
        
        if model_data_toml !== nothing
            total_params = get(get(model_data_toml, "weights_summary", Dict()), "total_parameters", 0)
            if total_params > 0
                model_size = get(get(model_data_toml, "weights_summary", Dict()), "model_size_mb", 0.0)
                println("   🔢 Parâmetros do modelo: $(total_params)")
                println("   💾 Tamanho estimado: $(model_size) MB")
            end
            examples = get(model_data_toml, "prediction_examples", [])
            if length(examples) > 0
                println("\n📏 Exemplo salvo como predição #$(length(examples))")
            end
        end
        
        # Avaliar confiança
        println("\n🎚️ Avaliação da Confiança:")
        if confidence >= 0.8
            println("   ✅ Identificação com ALTA confiança")
        elseif confidence >= 0.6
            println("   ⚠️ Identificação com MÉDIA confiança")
        elseif confidence >= 0.4
            println("   ⚠️ Identificação com BAIXA confiança - verifique manualmente")
        else
            println("   ❓ Identificação com MUITO BAIXA confiança - pessoa desconhecida?")
        end
        
        # Verificação final da lógica
        println("\n🔍 Verificação de Sanidade:")
        println("   - Total de classes no modelo: $(length(person_names))")
        println("   - Pessoa identificada está na posição correta? $(person_name in person_names ? "✅" : "❌")")
        if person_name in person_names
            correct_index = findfirst(x -> x == person_name, person_names)
            println("   - Índice correto de '$person_name': $correct_index")
        end
        
        return true
        
    catch e
        println("❌ Erro durante identificação: $e")
        println("📏 Tipo do erro: $(typeof(e))")
        if isa(e, MethodError)
            println("📏 Erro de método - possivelmente relacionado ao modelo ou dados")
        elseif isa(e, ArgumentError)
            println("📏 Erro de argumento - possivelmente relacionado a tipos ou conversões")
        end
        return false
    end
end

# Executar comando se for chamado diretamente
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        println("❌ Erro: Caminho da imagem não fornecido")
        println("Uso: julia cnncheckin_identify_fixed.jl <caminho_da_imagem>")
        println()
        println("Exemplo:")
        println("  julia cnncheckin_identify_fixed.jl ../../../dados/fotos_teste/teste.png")
    else
        success = identify_command(ARGS[1])
        if success
            println("✅ Identificação concluída!")
        else
            println("💥 Falha na identificação")
            println("\n🔧 Dicas de troubleshooting:")
            println("   1. Verifique se a imagem existe e não está corrompida")
            println("   2. Execute: julia cnncheckin_validate.jl")
            println("   3. Se necessário, retreine o modelo: julia cnncheckin_train.jl")
        end
    end
end