# Quick fix for your immediate incremental learning issue
# File: quick_fix_incremental.jl

using Flux
using JLD2

include("cnncheckin_core.jl")
using .CNNCheckinCore

println("CORREÇÃO RÁPIDA DO MODELO CORROMPIDO")
println("="^50)

# Step 1: Backup current corrupted model
timestamp = replace(string(now()), ":" => "-")
backup_name = "face_recognition_model_corrupted_backup_$timestamp.jld2"

try
    cp("face_recognition_model.jld2", backup_name)
    println("Backup criado: $backup_name")
catch e
    println("Erro criando backup: $e")
end

# Step 2: Check what people we currently have
config = CNNCheckinCore.load_config("face_recognition_config.toml")
current_people = config["data"]["person_names"]
println("Pessoas atuais no modelo: $(join(current_people, ", "))")

# Step 3: Identify the issue - which are original vs new people?
println("\nQual foi o problema?")
println("O modelo está confundindo as classes porque os pesos foram corrompidos")
println("durante o treinamento incremental.")

# Step 4: Offer immediate solutions
println("\nSOLUÇÕES IMEDIATAS:")
println("="^30)

println("\n1. SOLUÇÃO RÁPIDA - Resetar pesos da classificação:")
println("   - Mantém features aprendidas")
println("   - Re-inicializa apenas camada final") 
println("   - Requer re-treino rápido")

println("\n2. SOLUÇÃO SEGURA - Voltar ao pré-treino:")
println("   - Usar backup do modelo pré-treinado")
println("   - Refazer incremental corretamente")
println("   - Mais demorado mas mais confiável")

print("\nEscolha uma opção (1 ou 2): ")

# For now, let's implement option 1 - quick reset
function quick_reset_final_layer()
    println("\nExecutando reset rápido da camada final...")
    
    try
        # Load current model
        data = load("face_recognition_model.jld2")
        model = data["model_data"]["model_state"]
        
        # Get model layers
        layers = collect(model)
        feature_layers = layers[1:end-1]
        old_final = layers[end]
        
        if !isa(old_final, Dense)
            error("Camada final não é Dense")
        end
        
        # Create new final layer with proper initialization
        input_size = size(old_final.weight, 2)
        num_classes = length(current_people)
        
        new_final = Dense(input_size, num_classes)
        
        # Initialize with Xavier/Glorot initialization
        scale = sqrt(2.0f0 / (input_size + num_classes))
        new_final.weight .= randn(Float32, num_classes, input_size) * scale
        new_final.bias .= zeros(Float32, num_classes)
        
        # Create new model
        new_model = Chain(feature_layers..., new_final)
        
        # Save reset model
        model_data = Dict(
            "model_state" => new_model,
            "person_names" => current_people,
            "model_type" => "reset_final_layer",
            "timestamp" => string(now()),
            "reset_reason" => "Corrupted incremental weights"
        )
        
        jldsave("face_recognition_model.jld2"; model_data=model_data)
        
        println("Modelo resetado salvo!")
        println("IMPORTANTE: Agora você precisa re-treinar o modelo:")
        println("  julia cnncheckin_incremental.jl")
        
        return true
        
    catch e
        println("Erro no reset: $e")
        return false
    end
end

function find_pretrained_backup()
    println("\nProcurando backup pré-treinado...")
    
    # Look for backup files
    all_files = readdir(".")
    pretrained_files = filter(f -> contains(f, "pretrained") && endswith(f, ".jld2"), all_files)
    
    if length(pretrained_files) > 0
        println("Backups encontrados:")
        for (i, file) in enumerate(pretrained_files)
            println("  $i. $file")
        end
        
        latest = sort(pretrained_files)[end]
        println("\nUsando o mais recente: $latest")
        
        try
            cp(latest, "face_recognition_model.jld2", force=true)
            
            # Load to check people
            data = load(latest)
            original_people = data["model_data"]["person_names"]
            println("Modelo restaurado com pessoas: $(join(original_people, ", "))")
            
            # Update config
            config["data"]["person_names"] = original_people
            config["model"]["num_classes"] = length(original_people)
            CNNCheckinCore.save_config(config, "face_recognition_config.toml")
            
            println("Agora execute o incremental corretamente:")
            println("  julia cnncheckin_incremental.jl")
            
            return true
        catch e
            println("Erro restaurando backup: $e")
            return false
        end
    else
        println("Nenhum backup pré-treinado encontrado")
        println("Você terá que re-treinar do zero:")
        println("  julia cnncheckin_pretrain.jl")
        return false
    end
end

# Execute based on choice (for now, let's do option 2 - safer)
println("\nExecutando SOLUÇÃO SEGURA (opção 2)...")
success = find_pretrained_backup()

if !success
    println("\nFallback para SOLUÇÃO RÁPIDA (opção 1)...")
    quick_reset_final_layer()
end

println("\n" * "="^50)
println("IMPORTANTE - PRÓXIMOS PASSOS:")
println("="^50)
println("1. Verifique se suas imagens incrementais estão corretas:")
println("   - Diretório: $(CNNCheckinCore.INCREMENTAL_DATA_PATH)")
println("   - Nomes dos arquivos: pessoa-numero.jpg")
println("   - Apenas PESSOAS NOVAS (não treinadas antes)")

println("\n2. Execute o treinamento incremental:")
println("   julia cnncheckin_incremental.jl")

println("\n3. Teste o modelo corrigido:")
println("   julia cnncheckin_identify.jl ../../../dados/fotos_auth/objeto-3.jpeg")

println("\n4. Se ainda houver problemas:")
println("   - Verifique se objeto-3.jpeg é realmente uma foto de cachorro")
println("   - Se for cachorro, o modelo deveria predizer 'cachorro'")
println("   - Se estiver predizendo errado, há problema nos dados de treino")

println("\nDICA IMPORTANTE:")
println("O arquivo 'objeto-3.jpeg' parece ser um cachorro.")
println("Se o modelo está predizendo 'junior' para um cachorro,")
println("isso indica que:")
println("1. Os pesos foram corrompidos no incremental, OU")
println("2. As imagens de treino estão mal rotuladas, OU")
println("3. O arquivo não é realmente um cachorro")

println("\nVerifique visualmente o arquivo:")
println("  ../../../dados/fotos_auth/objeto-3.jpeg")

# julia quick_fix_incremental.jl