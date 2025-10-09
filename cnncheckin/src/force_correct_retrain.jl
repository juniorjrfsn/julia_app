# Force Correct Retrain - Fix the recognition issue
# File: force_correct_retrain.jl

using Dates

println("🔧 CORREÇÃO FORÇADA E RE-TREINO")
println("="^70)

function create_backup()
    println("\n💾 CRIANDO BACKUP...")
    
    timestamp = replace(string(now()), ":" => "-", "." => "-")
    files_to_backup = [
        "face_recognition_model.jld2",
        "face_recognition_config.toml",
        "face_recognition_model_data.toml"
    ]
    
    backup_dir = "backup_before_retrain_$timestamp"
    
    try
        mkdir(backup_dir)
        
        for file in files_to_backup
            if isfile(file)
                cp(file, joinpath(backup_dir, file))
                println("   ✅ Backup: $file")
            end
        end
        
        println("   ✅ Backup completo em: $backup_dir")
        return true
    catch e
        println("   ⚠️  Erro no backup: $e")
        return false
    end
end

function analyze_problem()
    println("\n🔍 ANALISANDO O PROBLEMA...")
    
    println("\n   📊 Situação atual:")
    println("      - Modelo identifica objeto-3.jpeg como 'junior' (85.7%)")
    println("      - Deveria identificar como 'cachorro'")
    println()
    println("   🎯 Causas possíveis (em ordem de probabilidade):")
    println("      1. ⚠️  Imagens de treino mal rotuladas")
    println("         → Fotos de cachorro estão na pasta/arquivos do junior")
    println("      2. ⚠️  Modelo foi treinado com dados incorretos")
    println("      3. ⚠️  Pesos corrompidos no treinamento incremental")
    println()
    println("   💡 Solução: Verificar dados E re-treinar")
end

function check_training_images()
    println("\n📁 VERIFICANDO IMAGENS DE TREINO...")
    
    train_dir = "../../../dados/fotos_train"
    new_dir = "../../../dados/fotos_new"
    
    println("\n   Verificando: $train_dir")
    if isdir(train_dir)
        files = filter(f -> !startswith(f, "."), readdir(train_dir))
        println("   Arquivos encontrados: $(length(files))")
        
        # Group by person
        junior_files = filter(f -> startswith(f, "junior"), files)
        lele_files = filter(f -> startswith(f, "lele"), files)
        cachorro_files = filter(f -> startswith(f, "cachorro"), files)
        
        println("\n   📋 Distribuição:")
        println("      - junior: $(length(junior_files)) arquivos")
        if length(junior_files) > 0
            println("         Arquivos: $(join(junior_files[1:min(3, length(junior_files))], ", "))")
            if length(junior_files) > 3
                println("         ... e mais $(length(junior_files) - 3)")
            end
        end
        
        println("      - lele: $(length(lele_files)) arquivos")
        if length(lele_files) > 0
            println("         Arquivos: $(join(lele_files[1:min(3, length(lele_files))], ", "))")
            if length(lele_files) > 3
                println("         ... e mais $(length(lele_files) - 3)")
            end
        end
        
        println("      - cachorro: $(length(cachorro_files)) arquivos")
        if length(cachorro_files) > 0
            println("         Arquivos: $(join(cachorro_files[1:min(3, length(cachorro_files))], ", "))")
            if length(cachorro_files) > 3
                println("         ... e mais $(length(cachorro_files) - 3)")
            end
        end
        
        # Check for issues
        if length(cachorro_files) > 0 && length(junior_files) > 0
            println("\n   ⚠️  ATENÇÃO:")
            println("      Cachorro está em fotos_train/ junto com junior e lele")
            println("      Isso indica que:")
            println("      - Ou cachorro foi parte do treino inicial (OK)")
            println("      - Ou houve confusão nos arquivos (PROBLEMA)")
        end
    else
        println("   ❌ Diretório não encontrado!")
    end
    
    println("\n   Verificando: $new_dir")
    if isdir(new_dir)
        files = filter(f -> !startswith(f, "."), readdir(new_dir))
        println("   Arquivos encontrados: $(length(files))")
        
        if length(files) > 0
            # Group by person
            for file in files[1:min(5, length(files))]
                person = split(file, "-")[1]
                println("      - $file (pessoa: $person)")
            end
            if length(files) > 5
                println("      ... e mais $(length(files) - 5)")
            end
        end
    else
        println("   ❌ Diretório não encontrado!")
    end
end

function suggest_solution()
    println("\n" * "="^70)
    println("💡 SOLUÇÕES DISPONÍVEIS")
    println("="^70)
    
    println("\n🔍 OPÇÃO 1: INVESTIGAR DADOS (Recomendado primeiro)")
    println("   Execute: julia verify_training_data.jl")
    println("   Isso vai:")
    println("   - Listar todas as imagens por pessoa")
    println("   - Identificar arquivos mal rotulados")
    println("   - Gerar script de correção")
    println()
    
    println("🔧 OPÇÃO 2: RE-TREINAR DO ZERO (Solução definitiva)")
    println("   Comando: julia cnncheckin_pretrain.jl")
    println("   Isso vai:")
    println("   - Remover modelo atual")
    println("   - Treinar novo modelo com dados atuais")
    println("   - Criar modelo limpo sem corrupção")
    println()
    
    println("⚡ OPÇÃO 3: RE-TREINAR INCREMENTAL")
    println("   Comando: julia cnncheckin_incremental.jl")
    println("   Isso vai:")
    println("   - Manter base do modelo atual")
    println("   - Adicionar novas classes")
    println("   - Mais rápido mas pode manter erros")
    println()
    
    println("🎯 OPÇÃO 4: CORREÇÃO AUTOMÁTICA COMPLETA")
    println("   Comando: bash fix_all.sh")
    println("   Isso vai:")
    println("   - Verificar dados")
    println("   - Fazer backup")
    println("   - Re-treinar automaticamente")
    println()
end

function create_complete_fix_script()
    println("\n📝 GERANDO SCRIPT DE CORREÇÃO COMPLETA...")
    
    script = """#!/bin/bash
# Script de correção completa
# Gerado em: $(now())

set -e  # Exit on error

echo "🔧 INICIANDO CORREÇÃO COMPLETA"
echo "======================================================================"

# Step 1: Backup
echo ""
echo "📦 Passo 1: Criando backup..."
BACKUP_DIR="backup_\$(date +%Y%m%d_%H%M%S)"
mkdir -p "\$BACKUP_DIR"

if [ -f "face_recognition_model.jld2" ]; then
    cp face_recognition_model.jld2 "\$BACKUP_DIR/"
    echo "✅ Backup do modelo criado"
fi

if [ -f "face_recognition_config.toml" ]; then
    cp face_recognition_config.toml "\$BACKUP_DIR/"
    echo "✅ Backup do config criado"
fi

# Step 2: Verify data
echo ""
echo "🔍 Passo 2: Verificando dados de treino..."
echo "Por favor, verifique MANUALMENTE as seguintes imagens:"
echo ""

# List training images
echo "📁 TREINO INICIAL (../../../dados/fotos_train/):"
if [ -d "../../../dados/fotos_train" ]; then
    for file in ../../../dados/fotos_train/junior-*.{jpg,jpeg,png} 2>/dev/null; do
        if [ -f "\$file" ]; then
            echo "   ⚠️  VERIFIQUE: \$file"
            echo "      → Esta é realmente uma foto do Junior (pessoa)?"
        fi
    done
    
    for file in ../../../dados/fotos_train/cachorro-*.{jpg,jpeg,png} 2>/dev/null; do
        if [ -f "\$file" ]; then
            echo "   ⚠️  VERIFIQUE: \$file"
            echo "      → Esta é realmente uma foto de cachorro?"
        fi
    done
fi

echo ""
echo "📁 TREINO INCREMENTAL (../../../dados/fotos_new/):"
if [ -d "../../../dados/fotos_new" ]; then
    for file in ../../../dados/fotos_new/*.{jpg,jpeg,png} 2>/dev/null; do
        if [ -f "\$file" ]; then
            echo "   ⚠️  VERIFIQUE: \$file"
        fi
    done
fi

echo ""
echo "======================================================================"
echo "⏸️  PAUSA PARA VERIFICAÇÃO MANUAL"
echo "======================================================================"
echo ""
echo "Você verificou as imagens acima e corrigiu os erros?"
echo "Digite 'sim' para continuar com o re-treino ou 'nao' para cancelar:"
read -p "> " RESPONSE

if [ "\$RESPONSE" != "sim" ]; then
    echo "❌ Operação cancelada"
    echo "Corrija os arquivos manualmente e execute novamente"
    exit 1
fi

# Step 3: Clean old model
echo ""
echo "🧹 Passo 3: Removendo modelo antigo..."
if [ -f "face_recognition_model.jld2" ]; then
    rm face_recognition_model.jld2
    echo "✅ Modelo antigo removido"
fi

# Step 4: Retrain
echo ""
echo "🚀 Passo 4: Re-treinando modelo do zero..."
julia cnncheckin_pretrain.jl

if [ \$? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✅ RE-TREINO CONCLUÍDO COM SUCESSO!"
    echo "======================================================================"
    echo ""
    echo "🧪 Passo 5: Testando modelo..."
    
    # Test with the problematic image
    if [ -f "../../../dados/fotos_auth/objeto-3.jpeg" ]; then
        julia cnncheckin_identify.jl ../../../dados/fotos_auth/objeto-3.jpeg
    fi
    
    echo ""
    echo "📋 Próximos passos:"
    echo "   1. Verifique se objeto-3.jpeg agora é identificado corretamente"
    echo "   2. Se ainda houver erro, o problema está nos DADOS DE TREINO"
    echo "   3. Abra CADA imagem manualmente e verifique o rótulo"
    
else
    echo ""
    echo "======================================================================"
    echo "❌ ERRO NO RE-TREINO"
    echo "======================================================================"
    echo ""
    echo "Backup disponível em: \$BACKUP_DIR"
    echo "Para restaurar:"
    echo "  cp \$BACKUP_DIR/face_recognition_model.jld2 ."
    echo "  cp \$BACKUP_DIR/face_recognition_config.toml ."
    exit 1
fi
"""
    
    filename = "fix_all.sh"
    open(filename, "w") do io
        write(io, script)
    end
    
    try
        chmod(filename, 0o755)
        println("   ✅ Script criado: $filename")
        println("   Execute com: bash $filename")
        return true
    catch e
        println("   ❌ Erro: $e")
        return false
    end
end

function manual_check_guide()
    println("\n" * "="^70)
    println("📖 GUIA DE VERIFICAÇÃO MANUAL")
    println("="^70)
    
    println("\n🔍 Como verificar as imagens manualmente:")
    println()
    println("1. Abra cada arquivo de imagem")
    println("2. Confirme visualmente o conteúdo")
    println("3. Compare com o nome do arquivo")
    println()
    println("Exemplo:")
    println("   📄 junior-1.jpg")
    println("   ↓")
    println("   👨 Deve conter foto do Junior (pessoa)")
    println("   ❌ Se contém cachorro → ERRO!")
    println()
    println("   📄 cachorro-1.jpg")
    println("   ↓")
    println("   🐕 Deve conter foto de cachorro")
    println("   ❌ Se contém pessoa → ERRO!")
    println()
    
    println("🔧 Como corrigir:")
    println()
    println("   Se junior-5.jpg contém cachorro:")
    println("   $ cd ../../../dados/fotos_train")
    println("   $ mv junior-5.jpg cachorro-5.jpg")
    println()
    println("   Se cachorro-3.jpg contém pessoa:")
    println("   $ cd ../../../dados/fotos_train")
    println("   $ mv cachorro-3.jpg junior-3.jpg")
    println()
end

function show_specific_problem()
    println("\n" * "="^70)
    println("🎯 PROBLEMA ESPECÍFICO DETECTADO")
    println("="^70)
    
    println("\n❗ O modelo está identificando objeto-3.jpeg como 'junior'")
    println("   quando deveria ser 'cachorro'")
    println()
    println("   Isso significa UMA destas situações:")
    println()
    println("   1. 🖼️  objeto-3.jpeg NÃO é um cachorro")
    println("      → Verifique o arquivo manualmente")
    println("      → Se for pessoa, renomeie para o nome correto")
    println()
    println("   2. 📁 As imagens de treino de 'junior' contêm cachorros")
    println("      → Verifique TODAS as imagens junior-*.jpg")
    println("      → Mova imagens de cachorro para cachorro-*.jpg")
    println()
    println("   3. 📁 As imagens de treino de 'cachorro' estão mal rotuladas")
    println("      → Verifique TODAS as imagens cachorro-*.jpg")
    println("      → Se houver pessoas, mova para o nome correto")
    println()
    println("   4. ⚙️  O modelo foi treinado com dados incorretos")
    println("      → Corrija os dados (opções 1-3 acima)")
    println("      → Re-treine: julia cnncheckin_pretrain.jl")
    println()
end

# Main execution
function main()
    analyze_problem()
    check_training_images()
    show_specific_problem()
    manual_check_guide()
    suggest_solution()
    
    # Create backup
    create_backup()
    
    # Create fix script
    create_complete_fix_script()
    
    println("\n" * "="^70)
    println("✅ ANÁLISE COMPLETA")
    println("="^70)
    
    println("\n📋 PRÓXIMOS PASSOS RECOMENDADOS:")
    println()
    println("   1️⃣  VERIFICAR DADOS (MAIS IMPORTANTE!):")
    println("      julia verify_training_data.jl")
    println()
    println("   2️⃣  VERIFICAR IMAGEM ESPECÍFICA:")
    println("      # Abra e veja o que realmente é:")
    println("      xdg-open ../../../dados/fotos_auth/objeto-3.jpeg")
    println()
    println("   3️⃣  RE-TREINAR:")
    println("      bash fix_all.sh")
    println()
    println("   Ou manualmente:")
    println("      julia cnncheckin_pretrain.jl")
    println()
    
    println("\n⚠️  LEMBRE-SE:")
    println("   O modelo aprende o que você ensina!")
    println("   Se ensinar errado (dados mal rotulados), vai aprender errado.")
    println()
end

# Run
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end