#!/usr/bin/env julia
# Projeto: perceptronxor
# Arquivo: src/xor/xor.jl
#
# Rede neural 2 -> N -> 1 que aprende a porta lógica XOR (verdadeiro/falso).
#
# Uso:
#   julia xor.jl train [--epochs 20000] [--lr 0.5] [--hidden 4] [--seed 1234]
#                      [--tentativas 5] [--model xor_modelo.jls]
#   julia xor.jl infer A B [--model xor_modelo.jls]     (A e B: 0/1, v/f, true/false)
#   julia xor.jl interactive [--model xor_modelo.jls]
#   julia xor.jl help
#
# Exemplos:
#   julia xor.jl train
#   julia xor.jl infer 0 1      # -> VERDADEIRO
#   julia xor.jl infer 1 1      # -> FALSO

using Random
using Serialization
using Printf

const MODELO_PADRAO = "xor_modelo.jls"

# ---------------------------------------------------------------------------
# Dados (tabela verdade do XOR)
# ---------------------------------------------------------------------------
const X = Float64[0 0; 0 1; 1 0; 1 1]
const Y = Float64[0; 1; 1; 0]

# ---------------------------------------------------------------------------
# Rede neural
# ---------------------------------------------------------------------------
sigmoid(x) = 1 / (1 + exp(-x))
dsigmoid(a) = a * (1 - a)   # derivada em função da saída da sigmoide

struct Modelo
    W1::Matrix{Float64}
    b1::Matrix{Float64}
    W2::Matrix{Float64}
    b2::Matrix{Float64}
end

# Inicialização com pesos positivos e negativos (Xavier/Glorot)
function novo_modelo(entrada::Int, oculta::Int, saida::Int, rng)
    limite(n_in, n_out) = sqrt(6 / (n_in + n_out))
    glorot(n_in, n_out) = (rand(rng, n_in, n_out) .* 2 .- 1) .* limite(n_in, n_out)
    Modelo(glorot(entrada, oculta), zeros(1, oculta),
           glorot(oculta, saida), zeros(1, saida))
end

function forward(m::Modelo, x::AbstractMatrix)
    a1 = sigmoid.(x * m.W1 .+ m.b1)
    a2 = sigmoid.(a1 * m.W2 .+ m.b2)
    return a1, a2
end

prever_prob(m::Modelo, x1::Real, x2::Real) = forward(m, Float64[x1 x2])[2][1]

function precisao(m::Modelo)
    _, a2 = forward(m, X)
    return count(((a2 .>= 0.5) .== (Y .>= 0.5))) / length(Y)
end


# Propagação didática: imprime a soma ponderada e a sigmoide de cada neurônio.
function forward_didatico(m::Modelo, x1::Real, x2::Real; titulo="INFERÊNCIA")
    x = Float64[x1 x2]
    println("\n", "="^72)
    println("$titulo — PROPAGAÇÃO DIRETA (FORWARD)")
    println("="^72)
    @printf("Entradas: x₁ = %.0f, x₂ = %.0f\n", x1, x2)
    println("\nFórmula de ativação: σ(z) = 1 / (1 + exp(-z))")
    println("Cada neurônio calcula z = soma(entrada × peso) + viés; depois aplica σ(z).")

    a1 = zeros(1, size(m.W1, 2))
    for j in 1:size(m.W1, 2)
        termos = String[]
        for i in 1:size(m.W1, 1)
            push!(termos, @sprintf("(%.3f × %.3f)", x[i], m.W1[i,j]))
        end
        z = sum(x[1,i] * m.W1[i,j] for i in 1:size(m.W1,1)) + m.b1[1,j]
        a1[1,j] = sigmoid(z)
        println("\nCamada oculta — neurônio h$j:")
        println("  z$j = " * join(termos, " + ") * @sprintf(" + viés(%.3f) = %.6f", m.b1[1,j], z))
        println("  a$j = σ(z$j) = " * @sprintf("1 / (1 + exp(-%.6f)) = %.6f", z, a1[1,j]))
    end

    println("\nSaídas da camada oculta: ", round.(vec(a1); digits=6))
    termos_saida = String[]
    for j in 1:size(m.W2, 1)
        push!(termos_saida, @sprintf("(%.6f × %.6f)", a1[1,j], m.W2[j,1]))
    end
    z2 = sum(a1[1,j] * m.W2[j,1] for j in 1:size(m.W2,1)) + m.b2[1,1]
    y = sigmoid(z2)
    println("\nCamada de saída — neurônio y:")
    println("  z = " * join(termos_saida, " + ") * @sprintf(" + viés(%.6f) = %.6f", m.b2[1,1], z2))
    @printf("  y = σ(z) = 1 / (1 + exp(-%.6f)) = %.6f\n", z2, y)
    println("\nRegra de decisão: se y ≥ 0.5 → VERDADEIRO; caso contrário → FALSO.")
    @printf("  %.6f %s 0.5 → %s\n", y, y >= 0.5 ? "≥" : "<", rotulo(y >= 0.5))
    return a1, reshape([y], 1, 1)
end

# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------
function treinar!(m::Modelo; epochs, lr, tol=1e-4, log_cada=1000, didatico=false)
    n = size(X, 1)
    perda = Inf
    for epoch in 1:epochs
        a1, a2 = forward(m, X)

        perda = sum((Y .- a2) .^ 2) / n

        if didatico && epoch == 1
            println("\n" * "="^72)
            println("TREINAMENTO DIDÁTICO — ÉPOCA 1 (pesos antes da atualização)")
            println("="^72)
            for k in 1:size(X, 1)
                forward_didatico(m, X[k,1], X[k,2]; titulo="TREINO — exemplo $k")
                @printf("  Alvo esperado: y = %.0f\n", Y[k])
                @printf("  Erro do exemplo (alvo - saída): %.6f\n", Y[k] - a2[k,1])
            end
            @printf("\nPerda média quadrática usada pelo código: Loss = soma((Y - saída)^2) / n = %.8f\n", perda)
        end

        # Backpropagation: derivadas do erro em relação às ativações.
        d2 = (a2 .- Y) .* dsigmoid.(a2)
        d1 = (d2 * m.W2') .* dsigmoid.(a1)

        if didatico && epoch == 1
            println("\nBACKPROPAGATION — cálculo dos deltas")
            println("Derivada da sigmoide: σ'(a) = a × (1 - a)")
            println("Delta de saída: d2 = (saída - alvo) × saída × (1 - saída)")
            println("Delta oculto: d1 = (d2 × W2') × a1 × (1 - a1)")
            println("d2 por exemplo:"); display(d2)
            println("d1 por exemplo e neurônio oculto:"); display(d1)
            println("\nATUALIZAÇÃO DOS PARÂMETROS")
            println("W2 := W2 - lr × (a1' × d2) / n")
            println("b2 := b2 - lr × soma(d2) / n")
            println("W1 := W1 - lr × (X' × d1) / n")
            println("b1 := b1 - lr × soma(d1) / n")
            @printf("Taxa de aprendizado lr = %.4f; quantidade de exemplos n = %d\n", lr, n)
        end

        m.W2 .-= lr .* (a1' * d2) ./ n
        m.b2 .-= lr .* sum(d2, dims=1) ./ n
        m.W1 .-= lr .* (X' * d1) ./ n
        m.b1 .-= lr .* sum(d1, dims=1) ./ n

        if didatico && epoch == 1
            println("\nParâmetros após a primeira atualização:")
            println("W1 (entradas → camada oculta):"); display(m.W1)
            println("b1 (vieses da camada oculta):"); display(m.b1)
            println("W2 (camada oculta → saída):"); display(m.W2)
            println("b2 (viés da saída):"); display(m.b2)
            println("Observação: o treino é em lote; os gradientes são calculados com os 4 exemplos.")
        end

        if epoch % log_cada == 0 || epoch == 1
            @printf("  Época %6d - Loss: %.6f\n", epoch, perda)
        end
        if perda < tol
            @printf("  Convergiu na época %d (Loss: %.6f)\n", epoch, perda)
            break
        end
    end
    return perda
end

function cmd_train(opts)
    epochs      = parse(Int, get(opts, "epochs", "20000"))
    lr          = parse(Float64, get(opts, "lr", "0.5"))
    oculta      = parse(Int, get(opts, "hidden", "4"))
    seed        = parse(Int, get(opts, "seed", "1234"))
    tentativas  = parse(Int, get(opts, "tentativas", "5"))
    caminho     = get(opts, "model", MODELO_PADRAO)
    didatico    = get(opts, "didatico", "false") in ("1", "true", "sim", "yes")

    println("Treinando a rede 2 → $oculta → 1 (lr=$lr, épocas=$epochs)...")

    melhor = nothing
    melhor_prec = -1.0
    for t in 1:tentativas
        println("Tentativa $t/$tentativas (seed=$(seed + t - 1))")
        rng = MersenneTwister(seed + t - 1)
        m = novo_modelo(2, oculta, 1, rng)
        treinar!(m; epochs=epochs, lr=lr, didatico=didatico)
        p = precisao(m)
        if p > melhor_prec
            melhor, melhor_prec = m, p
        end
        if p == 1.0
            break
        end
        println("  Precisão $(round(100p; digits=1))% — tentando outra inicialização...")
    end

    serialize(caminho, melhor)
    println("\nResultados da porta lógica XOR:")
    mostrar_tabela(melhor)
    println("\nPrecisão final: $(round(100 * melhor_prec; digits=1))%")
    println("Modelo salvo em: $caminho")
    melhor_prec < 1.0 && println("Aviso: a rede não aprendeu 100% do XOR. Tente mais --epochs ou --hidden maior.")
end

# ---------------------------------------------------------------------------
# Inferência
# ---------------------------------------------------------------------------
function carregar_modelo(caminho)
    if !isfile(caminho)
        println(stderr, "Modelo '$caminho' não encontrado. Rode primeiro: julia xor.jl train")
        exit(1)
    end
    return deserialize(caminho)
end

function parse_bool(s::AbstractString)
    s = lowercase(strip(s))
    s in ("0", "f", "false", "falso") && return 0.0
    s in ("1", "v", "t", "true", "verdadeiro") && return 1.0
    return nothing
end

rotulo(b::Bool) = b ? "VERDADEIRO" : "FALSO"

function inferir(m::Modelo, x1, x2)
    p = prever_prob(m, x1, x2)
    return p, p >= 0.5
end

function mostrar_tabela(m::Modelo)
    for i in 1:size(X, 1)
        p, r = inferir(m, X[i, 1], X[i, 2])
        esperado = Y[i] >= 0.5
        ok = r == esperado ? "✓" : "✗"
        @printf("  %d XOR %d -> prob=%.3f -> %-10s | esperado: %-10s %s\n",
                Int(X[i, 1]), Int(X[i, 2]), p, rotulo(r), rotulo(esperado), ok)
    end
end

function cmd_infer(args, opts)
    if length(args) != 2
        println(stderr, "Uso: julia xor.jl infer A B   (ex.: julia xor.jl infer 0 1)")
        exit(1)
    end
    x1, x2 = parse_bool(args[1]), parse_bool(args[2])
    if x1 === nothing || x2 === nothing
        println(stderr, "Entrada inválida. Use 0/1, v/f ou true/false.")
        exit(1)
    end
    m = carregar_modelo(get(opts, "model", MODELO_PADRAO))
    if get(opts, "didatico", "true") in ("1", "true", "sim", "yes")
        _, saida = forward_didatico(m, x1, x2)
        p = saida[1,1]
        r = p >= 0.5
    else
        p, r = inferir(m, x1, x2)
    end
    @printf("\nRESULTADO: %d XOR %d = %s (probabilidade: %.3f)\n", Int(x1), Int(x2), rotulo(r), p)
end

function cmd_interactive(opts)
    m = carregar_modelo(get(opts, "model", MODELO_PADRAO))
    println("--- Modo Interativo --- (digite dois valores separados por espaço, ex.: 0 1; vazio para sair)")
    while true
        print("> ")
        linha = readline()
        isempty(strip(linha)) && (println("Encerrando..."); break)
        partes = split(linha)
        if length(partes) != 2
            println("Digite exatamente dois valores.")
            continue
        end
        x1, x2 = parse_bool(partes[1]), parse_bool(partes[2])
        if x1 === nothing || x2 === nothing
            println("Entrada inválida. Use 0/1, v/f ou true/false.")
            continue
        end
        p, r = inferir(m, x1, x2)
        @printf("%d XOR %d = %s (probabilidade: %.3f)\n", Int(x1), Int(x2), rotulo(r), p)
    end
end

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
function ajuda()
    println("""
    Uso:
      julia xor.jl train [--epochs N] [--lr X] [--hidden N] [--seed N] [--tentativas N] [--model ARQ] [--didatico true]
      julia xor.jl infer A B [--model ARQ] [--didatico true|false]
      julia xor.jl interactive [--model ARQ]
      julia xor.jl help
    """)
end

function parse_args(argv)
    posicionais = String[]
    opts = Dict{String,String}()
    i = 1
    while i <= length(argv)
        a = argv[i]
        if startswith(a, "--")
            if i == length(argv)
                println(stderr, "Opção $a precisa de um valor.")
                exit(1)
            end
            opts[a[3:end]] = argv[i + 1]
            i += 2
        else
            push!(posicionais, a)
            i += 1
        end
    end
    return posicionais, opts
end

function main(argv)
    pos, opts = parse_args(argv)
    cmd = isempty(pos) ? "help" : lowercase(pos[1])
    resto = pos[2:end]
    if cmd == "train"
        cmd_train(opts)
    elseif cmd == "infer"
        cmd_infer(resto, opts)
    elseif cmd == "interactive"
        cmd_interactive(opts)
    else
        ajuda()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end

# cd perceptronxor/src/xor
# julia xor.jl train                      # treina e salva xor_modelo.jls
# julia xor.jl train --epochs 30000 --lr 0.5 --hidden 6
# julia xor.jl infer 0 1                  # 0 XOR 1 = VERDADEIRO (probabilidade: 0.98x)
# julia xor.jl infer v v                  # aceita 0/1, v/f, true/false
# julia xor.jl interactive                # digite "0 1" por linha; vazio sai

# julia xor_didatico.jl train --didatico true
# julia xor_didatico.jl infer 0 1

# julia xor_didatico.jl infer 1 0
# julia xor_didatico.jl infer 1 1
# julia xor_didatico.jl infer 0 0

