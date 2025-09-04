#!/usr/bin/env julia
# projeto: webcamcnn
# file: webcamcnn/src/start_system.jl

# Script de inicialização simplificado para evitar problemas de dependência circular

println("🚀 Iniciando CNN Checkin System...")

# Verificar se todos os arquivos necessários existem
required_files = [
    "core.jl",
    "weights_manager.jl", 
    "weights_utils.jl",
    "pretrain_modified.jl",
    "main_toml_system.jl"
]

missing_files = String[]
for file in required_files
    if !isfile(file)
        push!(missing_files, file)
    end
end

if !isempty(missing_files)
    println("❌ Arquivos ausentes: $(join(missing_files, ", "))")
    println("Certifique-se de que todos os arquivos estão no diretório atual.")
    exit(1)
end

# Tentar carregar dependências
try
    using Pkg
    
    # Lista de pacotes necessários
    required_packages = [
        "Flux",
        "Images", 
        "FileIO",
        "CUDA",
        "Statistics",
        "Random",
        "JLD2",
        "TOML",
        "ImageTransformations",
        "LinearAlgebra",
        "Dates",
        "VideoIO",
        "ImageView",
        "SHA"
    ]
    
    println("🔍 Verificando dependências...")
    
    missing_packages = String[]
    for pkg in required_packages
        try
            eval(Meta.parse("using $pkg"))
        catch
            push!(missing_packages, pkg)
        end
    end
    
    if !isempty(missing_packages)
        println("📦 Instalando pacotes ausentes: $(join(missing_packages, ", "))")
        Pkg.add(missing_packages)
        
        # Tentar carregar novamente
        for pkg in missing_packages
            try
                eval(Meta.parse("using $pkg"))
                println("✅ $pkg carregado com sucesso")
            catch e
                println("❌ Falha ao carregar $pkg: $e")
            end
        end
    else
        println("✅ Todas as dependências estão disponíveis")
    end
    
catch e
    println("⚠️ Erro ao verificar dependências: $e")
    println("Continuando mesmo assim...")
end

# Incluir e executar o sistema principal
try
    include("main_toml_system.jl")
    
    println("\n🎬 Executando sistema principal...")
    success = main()
    
    if success
        println("✅ Sistema encerrado com sucesso!")
    else
        println("⚠️ Sistema encerrado com avisos")
    end
    
catch e
    println("❌ Erro ao executar sistema: $e")
    println("\n🔧 Tentando modo de recuperação...")
    
    # Modo de recuperação básico
    try
        include("capture_and_train.jl")
        println("📸 Usando modo de captura básico...")
        main()
    catch recovery_error
        println("❌ Modo de recuperação falhou: $recovery_error")
        println("\n📖 Soluções sugeridas:")
        println("1. Verifique se a webcam está conectada")
        println("2. Execute: julia -e 'using Pkg; Pkg.add([\"Flux\", \"Images\", \"VideoIO\", \"JLD2\", \"TOML\"])'")
        println("3. Reinicie o Julia e tente novamente")
        println("4. Verifique permissões de acesso à webcam")
    end
end