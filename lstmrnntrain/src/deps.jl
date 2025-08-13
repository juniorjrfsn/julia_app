#!/usr/bin/env julia
# projeto : lstmrnntrain
# file : lstmrnntrain/src/deps.jl
# Script para instalar todas as dependências necessárias

using Pkg

println("=== Instalação de Dependências para Treinamento LSTM/RNN ===\n")

# Lista de pacotes necessários
required_packages = [
    "Flux",           # Framework de deep learning
    "TOML",           # Para ler arquivos TOML
    "Statistics",     # Estatísticas básicas
    "Dates",          # Manipulação de datas
    "Random",         # Geração de números aleatórios
    "BSON",           # Serialização de objetos
    "StatsBase",      # Estatísticas avançadas
    "CUDA"            # Suporte para GPU (opcional)
]

println("Instalando pacotes necessários...")

for pkg in required_packages
    try
        println("Instalando $pkg...")
        Pkg.add(pkg)
        println("✓ $pkg instalado com sucesso")
    catch e
        println("⚠ Erro ao instalar $pkg: $e")
        if pkg == "CUDA"
            println("  CUDA é opcional - continue sem GPU se necessário")
        end
    end
end

println("\n=== Verificando Instalação ===")

# Testar importação dos pacotes
test_packages = [
    ("Flux", "using Flux"),
    ("TOML", "using TOML"),
    ("Statistics", "using Statistics"),
    ("BSON", "using BSON"),
    ("StatsBase", "using StatsBase")
]

all_ok = true

for (name, import_cmd) in test_packages
    try
        eval(Meta.parse(import_cmd))
        println("✓ $name - OK")
    catch e
        println("✗ $name - ERRO: $e")
        all_ok = false
    end
end

if all_ok
    println("\n✓ Todas as dependências foram instaladas com sucesso!")
    println("\nVocê pode agora executar o script de treinamento:")
    println("julia treinar_modelos.jl")
else
    println("\n⚠ Algumas dependências apresentaram problemas.")
    println("Por favor, verifique os erros acima e tente reinstalar os pacotes com falha.")
end

println("\n=== Informações do Sistema ===")
println("Versão Julia: $(VERSION)")
println("Arquitetura: $(Sys.MACHINE)")
println("Sistema: $(Sys.KERNEL)")

# Verificar se CUDA está disponível
try
    using CUDA
    if CUDA.functional()
        println("✓ CUDA disponível - Treinamento com GPU possível")
        println("GPU: $(CUDA.device())")
    else
        println("⚠ CUDA instalado mas não funcional - Usando CPU")
    end
catch
    println("ℹ CUDA não disponível - Usando CPU para treinamento")
end


# julia deps.jl