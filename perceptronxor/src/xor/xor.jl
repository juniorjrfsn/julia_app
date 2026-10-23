#!/usr/bin/env julia
# xor.jl — Rede neural MLP para aprender a porta lógica XOR
#
# Arquitetura padrão: 2 entradas -> 4 neurônios ocultos -> 1 saída.
# Dependências: apenas bibliotecas padrão do Julia.
#
# Uso:
#   julia xor.jl train
#   julia xor.jl train --epochs 30000 --lr 0.5 --hidden 4 --seed 1234
#   julia xor.jl infer 0 1
#   julia xor.jl infer 1 1 --model xor_modelo.jls
#   julia xor.jl interactive
#   julia xor.jl help

using Random
using Serialization
using Printf

const MODELO_PADRAO = "xor_modelo.jls"

# Tabela-verdade XOR: cada linha é um exemplo; Y é uma matriz n × 1.
const X = Float64[0 0; 0 1; 1 0; 1 1]
const Y = reshape(Float64[0, 1, 1, 0], :, 1)

sigmoid(z) = 1.0 ./ (1.0 .+ exp.(-z))
dsigmoid(a) = a .* (1.0 .- a)

struct Modelo
    W1::Matrix{Float64}  # entradas -> camada oculta
    b1::Matrix{Float64}
    W2::Matrix{Float64}  # camada oculta -> saída
    b2::Matrix{Float64}
end

function novo_modelo(nentrada::Int, noculta::Int, nsaida::Int, rng)
    glorot(nin, nout) = (2 .* rand(rng, nin, nout) .- 1) .* sqrt(6 / (nin + nout))
    return Modelo(glorot(nentrada, noculta), zeros(1, noculta),
                  glorot(noculta, nsaida), zeros(1, nsaida))
end

function forward(m::Modelo, x::AbstractMatrix)
    z1 = x * m.W1 .+ m.b1
    a1 = sigmoid(z1)
    z2 = a1 * m.W2 .+ m.b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2
end

function perda(m::Modelo)
    _, _, _, yhat = forward(m, X)
    return sum((Y .- yhat).^2) / size(X, 1)
end

function precisao(m::Modelo)
    _, _, _, yhat = forward(m, X)
    return count((yhat .>= 0.5) .== (Y .>= 0.5)) / length(Y)
end

function treinar!(m::Modelo; epochs::Int=30000, lr::Float64=0.5,
                  tol::Float64=1e-3, log_cada::Int=5000)
    n = size(X, 1)
    for epoch in 1:epochs
        _, a1, _, a2 = forward(m, X)

        # Backpropagation para erro quadrático médio.
        d2 = (a2 .- Y) .* dsigmoid(a2)
        d1 = (d2 * m.W2') .* dsigmoid(a1)

        # Atualização em lote: todos os quatro exemplos contribuem.
        m.W2 .-= lr .* (a1' * d2) ./ n
        m.b2 .-= lr .* sum(d2, dims=1) ./ n
        m.W1 .-= lr .* (X' * d1) ./ n
        m.b1 .-= lr .* sum(d1, dims=1) ./ n

        if epoch == 1 || epoch % log_cada == 0
            @printf("Época %6d | loss = %.8f | precisão = %.0f%%\n",
                    epoch, perda(m), 100 * precisao(m))
        end
        perda(m) < tol && return epoch, perda(m)
    end
    return epochs, perda(m)
end

rotulo(v::Bool) = v ? "VERDADEIRO" : "FALSO"

function parse_bool(s::AbstractString)
    t = lowercase(strip(s))
    t in ("0", "f", "false", "falso") && return 0.0
    t in ("1", "v", "t", "true", "verdadeiro") && return 1.0
    return nothing
end

function inferir(m::Modelo, x1::Real, x2::Real; didatico::Bool=true)
    x = reshape(Float64[x1, x2], 1, 2)
    z1, a1, z2, a2 = forward(m, x)
    p = a2[1, 1]
    if didatico
        println("\n", "="^64)
        println("INFERÊNCIA MLP XOR — PROPAGAÇÃO DIRETA (FORWARD)")
        println("="^64)
        @printf("Entrada: x₁ = %.0f, x₂ = %.0f\n", x1, x2)
        println("\nCamada oculta:")
        for j in axes(a1, 2)
            @printf("  h%d: z = % .6f  ->  a = σ(z) = %.6f\n", j, z1[1,j], a1[1,j])
        end
        @printf("\nCamada de saída: z = %.6f\n", z2[1,1])
        @printf("y = σ(z) = %.6f\n", p)
        println("Regra: y ≥ 0.5 → VERDADEIRO; caso contrário → FALSO.")
    end
    @printf("\nRESULTADO: %.0f XOR %.0f = %s (probabilidade: %.4f)\n",
            x1, x2, rotulo(p >= 0.5), p)
    return p, p >= 0.5
end

function mostrar_tabela(m::Modelo)
    println("\nTabela-verdade aprendida:")
    println(" A  B | Probabilidade | Predição     | Esperado |")
    println("-----+---------------+--------------+----------")
    for i in axes(X, 1)
        p = forward(m, X[i:i, :])[4][1,1]
        pred = p >= 0.5
        esperado = Y[i,1] >= 0.5
        @printf(" %d  %d |    %.6f   | %-12s | %-9s %s\n",
                Int(X[i,1]), Int(X[i,2]), p, rotulo(pred),
                rotulo(esperado), pred == esperado ? "✓" : "✗")
    end
end

function parse_opcoes(args)
    pos = String[]
    opts = Dict{String,String}()
    i = 1
    while i <= length(args)
        if startswith(args[i], "--")
            i == length(args) && error("A opção $(args[i]) precisa de um valor.")
            opts[args[i][3:end]] = args[i+1]
            i += 2
        else
            push!(pos, args[i])
            i += 1
        end
    end
    return pos, opts
end

function cmd_train(opts)
    epochs = parse(Int, get(opts, "epochs", "30000"))
    lr = parse(Float64, get(opts, "lr", "0.5"))
    hidden = parse(Int, get(opts, "hidden", "4"))
    seed = parse(Int, get(opts, "seed", "1234"))
    tentativas = parse(Int, get(opts, "tentativas", "10"))
    caminho = get(opts, "model", MODELO_PADRAO)
    tol = parse(Float64, get(opts, "tol", "0.001"))

    epochs > 0 || error("--epochs deve ser maior que zero.")
    lr > 0 || error("--lr deve ser maior que zero.")
    hidden > 0 || error("--hidden deve ser maior que zero.")
    tentativas > 0 || error("--tentativas deve ser maior que zero.")

    println("Treinando MLP 2 → $hidden → 1 | épocas=$epochs | taxa=$lr")
    melhor = nothing
    melhor_acc = -1.0
    melhor_loss = Inf

    for tentativa in 1:tentativas
        println("\nTentativa $tentativa/$tentativas (seed=$(seed + tentativa - 1))")
        m = novo_modelo(2, hidden, 1, MersenneTwister(seed + tentativa - 1))
        ep, l = treinar!(m; epochs=epochs, lr=lr, tol=tol)
        acc = precisao(m)
        if acc > melhor_acc || (acc == melhor_acc && l < melhor_loss)
            melhor, melhor_acc, melhor_loss = m, acc, l
        end
        @printf("Fim da tentativa: época=%d | loss=%.8f | precisão=%.0f%%\n", ep, l, 100acc)
        acc == 1.0 && l < 0.02 && break
    end

    serialize(caminho, melhor)
    mostrar_tabela(melhor)
    @printf("\nPrecisão final: %.0f%% | loss: %.8f\n", 100 * precisao(melhor), perda(melhor))
    println("Modelo salvo em: $caminho")
    precisao(melhor) < 1.0 && println("Aviso: tente aumentar --epochs ou --tentativas.")
end

function carregar_modelo(caminho)
    isfile(caminho) || error("Modelo '$caminho' não encontrado. Execute: julia xor.jl train")
    m = deserialize(caminho)
    m isa Modelo || error("O arquivo não contém um modelo MLP compatível.")
    return m
end

function cmd_infer(resto, opts)
    length(resto) == 2 || error("Uso: julia xor.jl infer A B [--model arquivo]")
    x1, x2 = parse_bool(resto[1]), parse_bool(resto[2])
    (x1 === nothing || x2 === nothing) && error("Entrada inválida. Use 0/1, v/f ou true/false.")
    m = carregar_modelo(get(opts, "model", MODELO_PADRAO))
    inferir(m, x1, x2; didatico=get(opts, "didatico", "true") != "false")
end

function cmd_interactive(opts)
    m = carregar_modelo(get(opts, "model", MODELO_PADRAO))
    println("Modo interativo — informe dois valores (0/1, v/f); linha vazia encerra.")
    while true
        print("> ")
        linha = readline()
        isempty(strip(linha)) && break
        partes = split(linha)
        if length(partes) != 2
            println("Digite exatamente dois valores, por exemplo: 0 1")
            continue
        end
        x1, x2 = parse_bool(partes[1]), parse_bool(partes[2])
        if x1 === nothing || x2 === nothing
            println("Entrada inválida. Use 0/1, v/f ou true/false.")
            continue
        end
        inferir(m, x1, x2; didatico=false)
    end
end

function ajuda()
    println("""
    MLP XOR — xor.jl

    Comandos:
      julia xor.jl train [--epochs N] [--lr X] [--hidden N] [--seed N]
                         [--tentativas N] [--tol X] [--model ARQUIVO]
      julia xor.jl infer A B [--model ARQUIVO] [--didatico true|false]
      julia xor.jl interactive [--model ARQUIVO]
      julia xor.jl help

    Exemplos:
      julia xor.jl train
      julia xor.jl infer 0 1
      julia xor.jl infer 1 1
      julia xor.jl interactive
    """)
end

function main(argv)
    try
        pos, opts = parse_opcoes(argv)
        comando = isempty(pos) ? "help" : lowercase(pos[1])
        resto = pos[2:end]
        if comando == "train"
            cmd_train(opts)
        elseif comando == "infer"
            cmd_infer(resto, opts)
        elseif comando == "interactive"
            cmd_interactive(opts)
        elseif comando == "help" || comando == "--help" || comando == "-h"
            ajuda()
        else
            error("Comando desconhecido: $comando. Execute 'julia xor.jl help'.")
        end
    catch err
        println(stderr, "Erro: ", sprint(showerror, err))
        exit(1)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end


# Características da rede neural
# - Arquitetura: 2 entradas → 4 neurônios na camada oculta → 1 saída.
# - Ativação: função sigmoide.
# - Treinamento: backpropagation com atualização dos pesos em lote.
# - Inicialização: método Glorot/Xavier.
# - Persistência: salva o modelo treinado em xor_modelo.jls.
# - Inferência: apresenta os cálculos intermediários e a probabilidade prevista.
# - Modo interativo: permite testar diferentes entradas sem reiniciar o programa.
# - Dependências: somente bibliotecas padrão do Julia.
# Como executar
# Salve xor.jl na pasta do seu projeto e execute os comandos no terminal.

# 1. Treinar a rede neural
# julia xor.jl train


# 2. Testar a entrada 0 1
# julia xor.jl infer 0 1


# 3. Testar as quatro combinações
# julia xor.jl infer 0 0
# julia xor.jl infer 0 1
# julia xor.jl infer 1 0
# julia xor.jl infer 1 1


# 4. Iniciar o modo interativo
# julia xor.jl interactive


# 5. Personalizar o treinamento
# julia xor.jl train --epochs 30000 --lr 0.5 --hidden 4 --seed 1234


# Resultado esperado
# Entrada	XOR esperado
# 0 0	FALSO
# 0 1	VERDADEIRO
# 1 0	VERDADEIRO
# 1 1	FALSO

# A rede utiliza um limiar de classificação de 0.5. Como o treinamento é baseado em inicialização aleatória e otimização iterativa, os resultados precisam ser verificados após o treinamento; não é garantido que toda execução aprenda corretamente as quatro combinações.

# Importante: o arquivo foi criado, mas não executei o treinamento em um ambiente Julia nesta sessão. Portanto, os resultados acima representam a tabela-verdade que o modelo deve aprender, não uma validação de execução.
