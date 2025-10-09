# Complete Diagnostic and Fix Script
# File: diagnose_and_fix.jl
# Run this to identify and fix recognition problems

using Flux
using JLD2
using TOML
using Statistics
using Images
using FileIO

include("cnncheckin_core.jl")
using .CNNCheckinCore

println("🔍 DIAGNÓSTICO COMPLETO DO SISTEMA DE RECONHECIMENTO")
println("="^70)

# Step 1: Check if model and config exist
function check_files()
    println("\n📁 VERIFICANDO ARQUIVOS...")
    
    files_ok = true
    
    if !isfile("face_recognition_model.jld2")
        println("❌ Modelo não encontrado: face_recognition_model.jld2")
        files_ok = false
    else
        println("✅ Modelo encontrado")
    end
    
    if !isfile("face_recognition_config.toml")
        println("❌ Config não encontrada: face_recognition_config.toml")
        files_ok = false
    else
        println("✅ Config encontrada")
    end
    
    return files_ok
end

# Step 2: Analyze model structure
function analyze_model()
    println("\n🧠 ANALISANDO ESTRUTURA DO MODELO...")
    
    try
        data = load("face_recognition_model.jld2")
        model = data["model_data"]["model_state"]
        
        # Count layers
        layers = collect(model)
        println("   Total de camadas: $(length(layers))")
        
        # Check final layer
        final_layer = layers[end]
        if isa(final_layer, Dense)
            input_size = size(final_layer.weight, 2)
            output_size = size(final_layer.weight, 1)
            println("   Camada final: Dense($input_size → $output_size)")
            println("   Dimensões da camada final:")
            println("      - Peso: $(size(final_layer.weight))")
            println("      - Bias: $(size(final_layer.bias))")
            
            # Check weight statistics
            w = final_layer.weight
            println("   Estatísticas dos pesos:")
            println("      - Média: $(round(mean(w), digits=6))")
            println("      - Desvio padrão: $(round(std(w), digits=6))")
            println("      - Min: $(round(minimum(w), digits=6))")
            println("      - Max: $(round(maximum(w), digits=6))")
            
            # Check for NaN or Inf
            if any(isnan.(w)) || any(isinf.(w))
                println("   ❌ PROBLEMA: Pesos contêm NaN ou Inf!")
                return false, output_size
            end
            
            # Check if weights are too similar (not trained)
            if std(w) < 0.001
                println("   ⚠️  AVISO: Pesos muito similares - modelo pode não estar treinado")
            end
            
            return true, output_size
        else
            println("   ❌ PROBLEMA: Última camada não é Dense!")
            return false, 0
        end
        
    catch e
        println("   ❌ Erro ao analisar modelo: $e")
        return false, 0
    end
end

# Step 3: Check config consistency
function check_config()
    println("\n⚙️  VERIFICANDO CONFIGURAÇÃO...")
    
    try
        config = TOML.parsefile("face_recognition_config.toml")
        
        num_classes = config["model"]["num_classes"]
        person_names = config["data"]["person_names"]
        
        println("   Classes no config: $num_classes")
        println("   Pessoas no config: $(length(person_names))")
        println("   Lista de pessoas: $(join(person_names, ", "))")
        
        if num_classes != length(person_names)
            println("   ❌ PROBLEMA: Número de classes não corresponde à lista de pessoas!")
            println("      num_classes=$num_classes mas length(person_names)=$(length(person_names))")
            return false, person_names
        end
        
        # Check for duplicates
        if length(unique(person_names)) != length(person_names)
            println("   ❌ PROBLEMA: Nomes duplicados na lista de pessoas!")
            return false, person_names
        end
        
        println("   ✅ Configuração consistente")
        return true, person_names
        
    catch e
        println("   ❌ Erro ao ler config: $e")
        return false, String[]
    end
end

# Step 4: Test model with dummy input
function test_model_inference(model_output_size, person_names)
    println("\n🧪 TESTANDO INFERÊNCIA DO MODELO...")
    
    try
        data = load("face_recognition_model.jld2")
        model = data["model_data"]["model_state"]
        
        # Create dummy input
        test_input = randn(Float32, 128, 128, 3, 1)
        
        println("   Input shape: $(size(test_input))")
        
        # Run inference
        output = model(test_input)
        
        println("   Output shape: $(size(output))")
        println("   Output values (logits):")
        
        logits_vec = vec(output)
        for (i, (name, logit)) in enumerate(zip(person_names, logits_vec))
            println("      $i. $name: $(round(logit, digits=4))")
        end
        
        # Apply softmax
        probs = softmax(logits_vec)
        println("\n   Probabilidades após softmax:")
        for (i, (name, prob)) in enumerate(zip(person_names, probs))
            println("      $i. $name: $(round(prob*100, digits=2))%")
        end
        
        pred_idx = argmax(probs)
        println("\n   Predição: $(person_names[pred_idx]) (índice $pred_idx)")
        
        # Check if output size matches config
        if length(logits_vec) != length(person_names)
            println("   ❌ PROBLEMA: Output do modelo ($(length(logits_vec))) não corresponde ao número de pessoas ($(length(person_names)))")
            return false
        end
        
        println("   ✅ Modelo executa corretamente")
        return true
        
    catch e
        println("   ❌ Erro na inferência: $e")
        println("\n   Stack trace:")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
        return false
    end
end

# Step 5: Test with real image
function test_real_image(test_image_path::String)
    println("\n📸 TESTANDO COM IMAGEM REAL...")
    println("   Imagem: $test_image_path")
    
    if !isfile(test_image_path)
        println("   ❌ Imagem não encontrada!")
        return false
    end
    
    try
        # Load model and config
        data = load("face_recognition_model.jld2")
        model = data["model_data"]["model_state"]
        
        config = TOML.parsefile("face_recognition_config.toml")
        person_names = config["data"]["person_names"]
        
        # Preprocess image
        img_arrays = CNNCheckinCore.preprocess_image(test_image_path; augment=false)
        
        if img_arrays === nothing || length(img_arrays) == 0
            println("   ❌ Falha ao processar imagem")
            return false
        end
        
        img_array = img_arrays[1]
        img_tensor = reshape(img_array, size(img_array)..., 1)
        
        println("   Tensor shape: $(size(img_tensor))")
        
        # Run inference
        logits = model(img_tensor)
        logits_vec = Float32.(vec(logits))
        
        # Calculate probabilities manually
        max_logit = maximum(logits_vec)
        exp_logits = exp.(logits_vec .- max_logit)
        probs = exp_logits ./ sum(exp_logits)
        
        println("\n   Resultado da predição:")
        for (i, (name, prob)) in enumerate(zip(person_names, probs))
            bar = "█" ^ Int(round(prob * 50))
            println("      $i. $name: $(round(prob*100, digits=2))% $bar")
        end
        
        pred_idx = argmax(probs)
        confidence = probs[pred_idx]
        
        println("\n   ✅ Pessoa identificada: $(person_names[pred_idx])")
        println("   ✅ Confiança: $(round(confidence*100, digits=2))%")
        
        return true
        
    catch e
        println("   ❌ Erro ao testar imagem: $e")
        return false
    end
end

# Step 6: Identify specific problems
function identify_problems()
    println("\n🔍 IDENTIFICANDO PROBLEMAS ESPECÍFICOS...")
    
    problems = String[]
    
    try
        config = TOML.parsefile("face_recognition_config.toml")
        
        # Check if incremental training was done
        if haskey(config, "incremental_stats")
            println("   ℹ️  Treinamento incremental detectado")
            
            inc_stats = config["incremental_stats"]
            original_people = inc_stats["original_people"]
            new_people = inc_stats["new_people"]
            
            println("      - Pessoas originais: $(join(original_people, ", "))")
            println("      - Pessoas novas: $(join(new_people, ", "))")
            
            # Check if model was properly saved after incremental
            data = load("face_recognition_model.jld2")
            if data["model_data"]["model_type"] == "incremental"
                println("      - Tipo do modelo: incremental")
            else
                push!(problems, "Modelo não foi marcado como incremental após treinamento")
            end
        end
        
        # Check training accuracy
        final_acc = config["training"]["final_accuracy"]
        if final_acc < 0.7
            push!(problems, "Acurácia muito baixa: $(round(final_acc*100, digits=2))%")
        end
        
        # Check if model file is recent
        model_time = mtime("face_recognition_model.jld2")
        config_time = mtime("face_recognition_config.toml")
        
        if abs(model_time - config_time) > 300  # More than 5 minutes difference
            push!(problems, "Modelo e config têm timestamps muito diferentes (possível dessincronia)")
        end
        
    catch e
        push!(problems, "Erro ao verificar configuração: $e")
    end
    
    if length(problems) > 0
        println("\n   ⚠️  Problemas encontrados:")
        for (i, problem) in enumerate(problems)
            println("      $i. $problem")
        end
    else
        println("   ✅ Nenhum problema óbvio detectado")
    end
    
    return problems
end

# Step 7: Suggest fixes
function suggest_fixes(problems)
    println("\n💡 SUGESTÕES DE CORREÇÃO...")
    println("="^70)
    
    if length(problems) == 0
        println("Modelo parece OK. Se ainda há erro de reconhecimento, pode ser:")
        println("1. Dados de treino ruins (imagens incorretas ou mal rotuladas)")
        println("2. Imagem de teste muito diferente das de treino")
        println("3. Pessoa na imagem não está no conjunto de treino")
    else
        println("Baseado nos problemas encontrados, recomendo:")
        println()
        
        has_dimension_problem = any(p -> contains(p, "corresponde") || contains(p, "dimensão"), problems)
        has_training_problem = any(p -> contains(p, "Acurácia") || contains(p, "treinado"), problems)
        has_incremental_problem = any(p -> contains(p, "incremental"), problems)
        
        if has_dimension_problem
            println("🔧 CORREÇÃO 1: Problema de dimensões")
            println("   Execute: julia rebuild_model.jl")
            println()
        end
        
        if has_training_problem
            println("🔧 CORREÇÃO 2: Retreinar o modelo")
            println("   1. Faça backup: cp face_recognition_model.jld2 backup_$(now()).jld2")
            println("   2. Execute: julia cnncheckin_pretrain.jl")
            println()
        end
        
        if has_incremental_problem
            println("🔧 CORREÇÃO 3: Corrigir treinamento incremental")
            println("   Use o script: julia incremental_fix_patch.jl")
            println()
        end
    end
    
    println("\n📋 CHECKLIST DE VERIFICAÇÃO MANUAL:")
    println("   □ Imagens de treino estão corretas e bem rotuladas?")
    println("   □ Nomes dos arquivos seguem padrão: nome-numero.extensao?")
    println("   □ Cada pessoa tem pelo menos 5-10 imagens diferentes?")
    println("   □ Imagens têm boa qualidade (não borradas, bem iluminadas)?")
    println("   □ Pessoa a ser reconhecida está no conjunto de treino?")
end

# Main diagnostic routine
function run_full_diagnostic(test_image::String = "")
    println("\n🚀 INICIANDO DIAGNÓSTICO COMPLETO...\n")
    
    # Step 1: Check files
    if !check_files()
        println("\n❌ ERRO CRÍTICO: Arquivos essenciais não encontrados!")
        println("Execute primeiro: julia cnncheckin_pretrain.jl")
        return false
    end
    
    # Step 2: Analyze model
    model_ok, output_size = analyze_model()
    
    # Step 3: Check config
    config_ok, person_names = check_config()
    
    # Step 4: Test inference
    inference_ok = false
    if model_ok && config_ok
        inference_ok = test_model_inference(output_size, person_names)
    end
    
    # Step 5: Test with real image if provided
    if !isempty(test_image) && inference_ok
        test_real_image(test_image)
    end
    
    # Step 6: Identify problems
    problems = identify_problems()
    
    # Step 7: Suggest fixes
    suggest_fixes(problems)
    
    println("\n" * "="^70)
    println("📊 RESUMO DO DIAGNÓSTICO")
    println("="^70)
    println("   Arquivos: $(check_files() ? "✅" : "❌")")
    println("   Estrutura do modelo: $(model_ok ? "✅" : "❌")")
    println("   Configuração: $(config_ok ? "✅" : "❌")")
    println("   Inferência: $(inference_ok ? "✅" : "❌")")
    println("   Problemas encontrados: $(length(problems))")
    println("="^70)
    
    return model_ok && config_ok && inference_ok
end

# Command line interface
if length(ARGS) == 0
    println("Uso:")
    println("  julia diagnose_and_fix.jl                    # Diagnóstico geral")
    println("  julia diagnose_and_fix.jl <caminho_imagem>   # Diagnóstico + teste com imagem")
    println()
    
    run_full_diagnostic()
else
    test_image = ARGS[1]
    run_full_diagnostic(test_image)
end