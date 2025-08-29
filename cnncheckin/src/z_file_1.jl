# Script para instalar e configurar pacotes necessários para CNN Check-In
# Execute este script no Julia REPL para instalar todas as dependências

using Pkg

println("🔧 Instalando pacotes necessários para CNN Check-In...")

# Lista de pacotes necessários
packages_to_install = [
    "Gtk",
    "Cairo", 
    "VideoIO",
    "Images",
    "FileIO",
    "Dates",
    "ImageIO"
]

# Instalar pacotes um por um com tratamento de erro
for pkg in packages_to_install
    try
        println("📦 Instalando $pkg...")
        Pkg.add(pkg)
        println("✅ $pkg instalado com sucesso")
    catch e
        println("⚠️ Erro ao instalar $pkg: $e")
    end
end

println("\n🔄 Reconstruindo pacotes...")
try
    Pkg.build()
    println("✅ Build concluído")
catch e
    println("⚠️ Aviso durante build: $e")
end

println("\n🧪 Testando importações...")
test_packages = [
    "Gtk" => "using Gtk",
    "Cairo" => "using Cairo", 
    "VideoIO" => "using VideoIO",
    "Images" => "using Images",
    "FileIO" => "using FileIO"
]

for (pkg_name, import_cmd) in test_packages
    try
        eval(Meta.parse(import_cmd))
        println("✅ $pkg_name - OK")
    catch e
        println("❌ $pkg_name - ERRO: $e")
    end
end

println("\n✅ Configuração concluída!")
println("💡 Agora execute: julia cnncheckin_acount.jl")


# julia z_file_1.jl

# z_file_2.sh

# z_file_3.sh