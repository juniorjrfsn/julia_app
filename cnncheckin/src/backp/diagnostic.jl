# diagnostic.jl - Script para diagnosticar problemas no sistema de reconhecimento facial

using Flux
using Images
using FileIO
using JLD2
using Statistics
using Dates  # Add this import for the now() function

# Incluir o módulo principal
include("cnncheckin.jl")
using .cnncheckin

const DATA_PATH = "../../../dados/fotos"
const MODEL_PATH = "face_recognition_model.jld2"
const TEST_IMAGE = "../../../dados/fotos_teste/534770020_18019526477744454_2931624826193581596_n.jpg"

function diagnose_training_data()
    println("🔍 DIAGNÓSTICO 1: Verificando dados de treinamento")
    println("="^60)
    
    if !isdir(DATA_PATH)
        println("❌ Diretório de dados não encontrado: $DATA_PATH")
        return false
    end
    
    # Listar arquivos
    files = readdir(DATA_PATH)
    image_files = filter(f -> lowercase(splitext(f)[2]) in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"], files)
    
    println("📁 Arquivos encontrados no diretório de treinamento:")
    for file in image_files
        person_name = cnncheckin.extract_person_name(file)
        println("  📄 $file → Pessoa: '$person_name'")
    end
    
    # Agrupar por pessoa
    person_counts = Dict{String, Int}()
    for file in image_files
        person_name = cnncheckin.extract_person_name(file)
        person_counts[person_name] = get(person_counts, person_name, 0) + 1
    end
    
    println("\n👥 Resumo por pessoa:")
    for (person, count) in sort(collect(person_counts))
        println("  👤 $person: $count imagens")
    end
    
    return true
end

function diagnose_model()
    println("\n🔍 DIAGNÓSTICO 2: Verificando modelo treinado")
    println("="^60)
    
    if !isfile(MODEL_PATH)
        println("❌ Modelo não encontrado: $MODEL_PATH")
        return false
    end
    
    try
        # Carregar modelo
        model, person_names = cnncheckin.load_model(MODEL_PATH)
        
        println("✅ Modelo carregado com sucesso")
        println("📊 Informações do modelo:")
        println("  🏷️  Número de classes: $(length(person_names))")
        println("  👥 Nomes das pessoas:")
        for (i, name) in enumerate(person_names)
            println("    $i. '$name'")
        end
        
        return true, model, person_names
    catch e
        println("❌ Erro ao carregar modelo: $e")
        return false, nothing, nothing
    end
end

function diagnose_test_image()
    println("\n🔍 DIAGNÓSTICO 3: Analisando imagem de teste")
    println("="^60)
    
    if !isfile(TEST_IMAGE)
        println("❌ Imagem de teste não encontrada: $TEST_IMAGE")
        return false
    end
    
    println("📸 Imagem de teste: $TEST_IMAGE")
    
    # Extrair nome esperado do arquivo (se seguir o padrão)
    filename = basename(TEST_IMAGE)
    if occursin("-", filename)
        expected_name = cnncheckin.extract_person_name(filename)
        println("🎯 Nome esperado (baseado no arquivo): '$expected_name'")
    else
        println("⚠️  Nome não pode ser extraído do arquivo (não segue padrão nome-xxx)")
    end
    
    # Tentar preprocessar a imagem
    try
        img_array = cnncheckin.preprocess_image(TEST_IMAGE)
        if img_array !== nothing
            println("✅ Imagem preprocessada com sucesso")
            println("  📏 Dimensões: $(size(img_array))")
            println("  📊 Valores min/max: $(minimum(img_array)) / $(maximum(img_array))")
        else
            println("❌ Falha no preprocessamento da imagem")
            return false
        end
    catch e
        println("❌ Erro no preprocessamento: $e")
        return false
    end
    
    return true
end

function diagnose_prediction_detailed()
    println("\n🔍 DIAGNÓSTICO 4: Análise detalhada da predição")
    println("="^60)
    
    # Carregar modelo
    success, model, person_names = diagnose_model()
    if !success
        return false
    end
    
    # Preprocessar imagem
    img_array = cnncheckin.preprocess_image(TEST_IMAGE)
    if img_array === nothing
        println("❌ Não foi possível processar a imagem")
        return false
    end
    
    # Adicionar dimensão de batch
    img_tensor = reshape(img_array, size(img_array)..., 1)
    
    try
        # Fazer predição
        prediction = model(img_tensor)
        pred_probs = vec(prediction)
        
        println("🎯 Probabilidades por classe:")
        for (i, prob) in enumerate(pred_probs)
            person_name = person_names[i]
            percentage = round(prob * 100, digits=2)
            println("  👤 $person_name: $percentage%")
        end
        
        # Classe predita
        pred_class = argmax(pred_probs)
        predicted_person = person_names[pred_class]
        confidence = pred_probs[pred_class]
        
        println("\n📊 Resultado:")
        println("  🏆 Pessoa predita: '$predicted_person'")
        println("  📈 Confiança: $(round(confidence*100, digits=2))%")
        
        # Verificar se todas as probabilidades estão concentradas em uma classe
        sorted_probs = sort(pred_probs, rev=true)
        if sorted_probs[1] > 0.99 && sorted_probs[2] < 0.01
            println("⚠️  PROBLEMA DETECTADO: Modelo muito confiante (possível overfitting)")
        end
        
        return true
    catch e
        println("❌ Erro na predição: $e")
        return false
    end
end

function suggest_solutions()
    println("\n💡 SUGESTÕES DE SOLUÇÃO")
    println("="^60)
    
    println("1. 📄 Verificar dados de treinamento:")
    println("   - Certifique-se de que os nomes dos arquivos estão corretos")
    println("   - Formato: 'pessoa-001.jpg', 'pessoa-002.jpg', etc.") 
    
    println("\n2. 🎯 Retreinar o modelo:")
    println("   - Execute: julia cnncheckin.jl --treino")
    println("   - Certifique-se de ter pelo menos 3-5 imagens por pessoa")
    
    println("\n3. 📊 Verificar qualidade dos dados:")
    println("   - As imagens devem ter rostos bem visíveis")
    println("   - Evite imagens muito escuras ou desfocadas")
    println("   - Tenha variedade de ângulos e iluminação")
    
    println("\n4. 🔍 Testar com outras imagens:")
    println("   - Teste com imagens diferentes da mesma pessoa")
    println("   - Teste com pessoas que estão no dataset de treinamento")
    
 
end

function run_full_diagnosis()
    println("🚀 DIAGNÓSTICO COMPLETO DO SISTEMA CNN CHECK-IN")
    println("="^60)
    println("⏰ $(now())")
    println()
    
    # Executar todos os diagnósticos
    diagnose_training_data()
    diagnose_model()
    diagnose_test_image()
    diagnose_prediction_detailed()
    suggest_solutions()
    
    println("\n" * "="^60)
    println("✅ Diagnóstico concluído!")
end

# Executar diagnóstico se for o arquivo principal
if abspath(PROGRAM_FILE) == @__FILE__
    run_full_diagnosis()
end


# julia diagnostic.jl