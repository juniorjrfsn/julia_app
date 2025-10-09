# Verify and Fix Training Data
# File: verify_training_data.jl
# This script checks if training images are correctly labeled

using Images
using FileIO
using Dates

println("🔍 VERIFICAÇÃO DE DADOS DE TREINO")
println("="^70)

function analyze_directory(dir_path::String, dir_name::String)
    println("\n📁 Analisando: $dir_name")
    println("   Caminho: $dir_path")
    
    if !isdir(dir_path)
        println("   ❌ Diretório não encontrado!")
        return Dict{String, Vector{String}}()
    end
    
    # Group files by person name
    person_files = Dict{String, Vector{String}}()
    
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"]
    
    files = readdir(dir_path)
    total_files = 0
    valid_images = 0
    invalid_files = 0
    
    for filename in files
        filepath = joinpath(dir_path, filename)
        
        # Check if it's a file
        if !isfile(filepath)
            continue
        end
        
        total_files += 1
        
        # Check extension
        ext = lowercase(splitext(filename)[2])
        if !(ext in image_extensions)
            println("   ⚠️  Arquivo não-imagem: $filename")
            invalid_files += 1
            continue
        end
        
        # Extract person name
        name_parts = split(splitext(filename)[1], "-")
        if length(name_parts) < 2
            println("   ⚠️  Nome incorreto (falta '-'): $filename")
            invalid_files += 1
            continue
        end
        
        person_name = name_parts[1]
        
        # Try to load image
        try
            img = load(filepath)
            valid_images += 1
            
            # Add to person's file list
            if !haskey(person_files, person_name)
                person_files[person_name] = String[]
            end
            push!(person_files[person_name], filename)
            
        catch e
            println("   ❌ Erro ao carregar $filename: $e")
            invalid_files += 1
        end
    end
    
    println("\n   📊 Resumo:")
    println("      Total de arquivos: $total_files")
    println("      Imagens válidas: $valid_images")
    println("      Arquivos inválidos: $invalid_files")
    
    # Show person breakdown
    if length(person_files) > 0
        println("\n   👥 Pessoas encontradas:")
        for (person, files) in sort(collect(person_files))
            println("      - $person: $(length(files)) imagens")
            if length(files) <= 10
                for file in files
                    println("         • $file")
                end
            else
                for file in files[1:3]
                    println("         • $file")
                end
                println("         • ... e mais $(length(files)-3)")
            end
        end
    end
    
    return person_files
end

function show_image_info(filepath::String)
    """Display information about a specific image"""
    if !isfile(filepath)
        println("   ❌ Arquivo não encontrado: $filepath")
        return
    end
    
    try
        img = load(filepath)
        
        println("\n   📷 Informações da imagem:")
        println("      Arquivo: $(basename(filepath))")
        println("      Dimensões: $(size(img))")
        println("      Tipo: $(typeof(img))")
        println("      Tamanho: $(round(stat(filepath).size / 1024, digits=2)) KB")
        
    catch e
        println("   ❌ Erro ao carregar: $e")
    end
end

function verify_all_directories()
    println("\n🔍 VERIFICANDO TODOS OS DIRETÓRIOS DE TREINO")
    println("="^70)
    
    # Define directories
    train_dir = "../../../dados/fotos_train"
    new_dir = "../../../dados/fotos_new"
    auth_dir = "../../../dados/fotos_auth"
    
    # Analyze each directory
    train_data = analyze_directory(train_dir, "TREINO INICIAL (fotos_train)")
    new_data = analyze_directory(new_dir, "TREINO INCREMENTAL (fotos_new)")
    auth_data = analyze_directory(auth_dir, "AUTENTICAÇÃO/TESTE (fotos_auth)")
    
    # Cross-check
    println("\n" * "="^70)
    println("🔍 ANÁLISE CRUZADA")
    println("="^70)
    
    all_people = Set{String}()
    union!(all_people, keys(train_data))
    union!(all_people, keys(new_data))
    
    println("\n👥 Todas as pessoas no sistema:")
    for person in sort(collect(all_people))
        in_train = haskey(train_data, person) ? length(train_data[person]) : 0
        in_new = haskey(new_data, person) ? length(new_data[person]) : 0
        total = in_train + in_new
        
        status = "✅"
        if total < 3
            status = "⚠️  POUCOS DADOS"
        end
        
        println("   $status $person:")
        println("      - Treino inicial: $in_train imagens")
        println("      - Treino incremental: $in_new imagens")
        println("      - Total: $total imagens")
    end
    
    # Check for problems
    println("\n" * "="^70)
    println("⚠️  PROBLEMAS POTENCIAIS")
    println("="^70)
    
    problems_found = false
    
    # Check if same person in both directories
    for person in keys(train_data)
        if haskey(new_data, person)
            println("\n   ⚠️  PROBLEMA: '$person' está em AMBOS os diretórios!")
            println("      - O incremental deve ter APENAS pessoas NOVAS")
            println("      - Remova as imagens de '$person' de fotos_new/")
            problems_found = true
        end
    end
    
    # Check for insufficient data
    for person in all_people
        in_train = haskey(train_data, person) ? length(train_data[person]) : 0
        in_new = haskey(new_data, person) ? length(new_data[person]) : 0
        total = in_train + in_new
        
        if total < 3
            println("\n   ⚠️  PROBLEMA: '$person' tem apenas $total imagens!")
            println("      - Mínimo recomendado: 5-10 imagens")
            println("      - Adicione mais imagens variadas")
            problems_found = true
        end
    end
    
    if !problems_found
        println("   ✅ Nenhum problema estrutural encontrado")
    end
    
    return train_data, new_data, auth_data, all_people
end

function suggest_reorganization(train_data, new_data, all_people)
    println("\n" * "="^70)
    println("💡 SUGESTÕES DE REORGANIZAÇÃO")
    println("="^70)
    
    # Check current model config
    if isfile("face_recognition_config.toml")
        using TOML
        config = TOML.parsefile("face_recognition_config.toml")
        model_people = config["data"]["person_names"]
        
        println("\n📋 Pessoas no modelo atual: $(join(model_people, ", "))")
        
        # Check if data matches model
        data_people = sort(collect(all_people))
        
        if Set(model_people) != Set(data_people)
            println("\n⚠️  DESCOMPASSO DETECTADO!")
            println("   Modelo tem: $(join(model_people, ", "))")
            println("   Dados têm: $(join(data_people, ", "))")
            
            extra_in_model = setdiff(Set(model_people), Set(data_people))
            extra_in_data = setdiff(Set(data_people), Set(model_people))
            
            if length(extra_in_model) > 0
                println("\n   ❌ Pessoas no modelo SEM dados: $(join(extra_in_model, ", "))")
            end
            
            if length(extra_in_data) > 0
                println("\n   ❌ Pessoas nos dados SEM treino: $(join(extra_in_data, ", "))")
            end
            
            println("\n   🔧 SOLUÇÃO: Re-treinar o modelo do zero")
            println("      julia cnncheckin_pretrain.jl")
        end
    end
    
    # Specific problem: cachorro being detected as junior
    println("\n" * "="^70)
    println("🐕 ANÁLISE ESPECÍFICA: Problema 'cachorro' vs 'junior'")
    println("="^70)
    
    println("\n❗ O modelo está confundindo cachorro com junior!")
    println("   Possíveis causas:")
    println()
    println("   1. ❌ Imagens de 'junior' contêm fotos de cachorro")
    println("      → Solução: Remova fotos de cachorro da pasta de junior")
    println()
    println("   2. ❌ Imagens de 'cachorro' estão rotuladas como 'junior'")
    println("      → Solução: Renomeie arquivos junior-X.jpg para cachorro-X.jpg")
    println()
    println("   3. ❌ Modelo foi treinado com dados incorretos")
    println("      → Solução: Corrija os dados E re-treine")
    println()
    
    println("   📋 AÇÃO IMEDIATA RECOMENDADA:")
    println("   1. Verifique MANUALMENTE cada imagem:")
    
    if haskey(train_data, "junior")
        println("\n      📁 fotos_train/:")
        for file in train_data["junior"]
            println("         ⚠️  Abra e verifique: ../../../dados/fotos_train/$file")
            println("            Esta é REALMENTE uma foto do Junior (pessoa)?")
        end
    end
    
    if haskey(new_data, "junior")
        println("\n      📁 fotos_new/:")
        for file in new_data["junior"]
            println("         ⚠️  Abra e verifique: ../../../dados/fotos_new/$file")
            println("            Esta é REALMENTE uma foto do Junior (pessoa)?")
        end
    end
    
    if haskey(train_data, "cachorro")
        println("\n      📁 fotos_train/:")
        for file in train_data["cachorro"]
            println("         ⚠️  Abra e verifique: ../../../dados/fotos_train/$file")
            println("            Esta é REALMENTE uma foto de cachorro?")
        end
    end
    
    if haskey(new_data, "cachorro")
        println("\n      📁 fotos_new/:")
        for file in new_data["cachorro"]
            println("         ⚠️  Abra e verifique: ../../../dados/fotos_new/$file")
            println("            Esta é REALMENTE uma foto de cachorro?")
        end
    end
    
    println("\n   2. Corrija arquivos mal rotulados:")
    println("      # Se junior-5.jpg é na verdade um cachorro:")
    println("      mv fotos_train/junior-5.jpg fotos_train/cachorro-5.jpg")
    println()
    println("   3. Após corrigir, re-treine:")
    println("      julia cnncheckin_pretrain.jl")
end

function interactive_verification()
    println("\n" * "="^70)
    println("🔍 VERIFICAÇÃO INTERATIVA")
    println("="^70)
    
    println("\nDeseja verificar arquivos específicos? (s/n): ")
    response = lowercase(strip(readline()))
    
    if response == "s" || response == "sim"
        while true
            println("\nDigite o caminho do arquivo (ou 'sair' para terminar):")
            print("> ")
            path = strip(readline())
            
            if lowercase(path) == "sair"
                break
            end
            
            if isfile(path)
                show_image_info(path)
                
                println("\n   Esta imagem está corretamente rotulada? (s/n/pular): ")
                answer = lowercase(strip(readline()))
                
                if answer == "n" || answer == "nao" || answer == "não"
                    println("   Qual deveria ser o rótulo correto?")
                    print("   > ")
                    correct_label = strip(readline())
                    
                    # Extract current number
                    filename = basename(path)
                    parts = split(splitext(filename)[1], "-")
                    if length(parts) >= 2
                        number = parts[end]
                        ext = splitext(filename)[2]
                        new_filename = "$(correct_label)-$(number)$(ext)"
                        new_path = joinpath(dirname(path), new_filename)
                        
                        println("\n   Sugestão de correção:")
                        println("      mv $path $new_path")
                        println("\n   Executar esta correção? (s/n): ")
                        confirm = lowercase(strip(readline()))
                        
                        if confirm == "s" || confirm == "sim"
                            try
                                mv(path, new_path)
                                println("   ✅ Arquivo renomeado!")
                            catch e
                                println("   ❌ Erro ao renomear: $e")
                            end
                        end
                    end
                end
            else
                println("   ❌ Arquivo não encontrado")
            end
        end
    end
end

function generate_correction_script(train_data, new_data)
    println("\n" * "="^70)
    println("📝 GERANDO SCRIPT DE CORREÇÃO")
    println("="^70)
    
    script_lines = String[
        "#!/bin/bash",
        "# Script de correção automática",
        "# Gerado em: $(Dates.now())",
        "",
        "echo '🔧 Corrigindo estrutura de dados...'",
        ""
    ]
    
    # Check for people in wrong directories
    original_people = ["junior", "lele"]  # From config
    
    for person in keys(new_data)
        if person in original_people
            push!(script_lines, "# PROBLEMA: '$person' está em fotos_new/ mas deveria estar apenas em fotos_train/")
            push!(script_lines, "echo '⚠️  Removendo $person de fotos_new/'")
            for file in new_data[person]
                push!(script_lines, "# rm ../../../dados/fotos_new/$file")
            end
            push!(script_lines, "")
        end
    end
    
    script_content = join(script_lines, "\n")
    
    script_file = "fix_training_data.sh"
    open(script_file, "w") do io
        write(io, script_content)
    end
    
    try
        chmod(script_file, 0o755)
    catch
    end
    
    println("\n   ✅ Script gerado: $script_file")
    println("   Execute com: bash $script_file")
end

# Main execution
function main()
    train_data, new_data, auth_data, all_people = verify_all_directories()
    
    suggest_reorganization(train_data, new_data, all_people)
    
    # Generate correction script
    generate_correction_script(train_data, new_data)
    
    # Offer interactive verification
    interactive_verification()
    
    println("\n" * "="^70)
    println("✅ VERIFICAÇÃO CONCLUÍDA")
    println("="^70)
    
    println("\n📋 PRÓXIMOS PASSOS:")
    println("   1. Corrija os arquivos identificados acima")
    println("   2. Execute: julia cnncheckin_pretrain.jl")
    println("   3. Teste: julia cnncheckin_identify.jl ../../../dados/fotos_auth/objeto-3.jpeg")
end

# Run
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end