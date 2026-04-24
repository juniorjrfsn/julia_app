# Projeto : chatbot
# Arquivo: src/chatbot.jl
#
# Chatbot com busca baseada em modelo TF-IDF pré-treinado.
# Execute treinar.jl primeiro para gerar o arquivo de modelo.
#
# Uso direto:
#   julia src/chatbot.jl
#   julia src/chatbot.jl data/modelo_treinado.toml

module chatbot

using TOML

# ─────────────────────────────────────────────
# Estrutura do modelo treinado
# ─────────────────────────────────────────────

struct ModeloTreinado
    vocab::Dict{String,Int}        # termo → índice
    idf::Vector{Float64}           # peso IDF por termo
    vetores::Vector{Vector{Float64}}  # vetores TF-IDF das perguntas (normalizados)
    par_idx::Vector{Int}           # vetor[k] → índice da resposta
    respostas::Vector{String}      # lista de respostas únicas
    padrao::String                 # resposta fallback
end

# ─────────────────────────────────────────────
# Pré-processamento (idêntico ao treinar.jl)
# ─────────────────────────────────────────────

function normalizar(texto::String)::String
    texto = lowercase(texto)
    texto = replace(texto, r"[^a-záàâãéèêíïóôõöúüç\s]" => "")
    return strip(texto)
end

function tokenizar(texto::String)::Vector{String}
    return String.(split(normalizar(texto)))
end

function ngramas(tokens::Vector{String}; n_max::Int=2)::Vector{String}
    result = copy(tokens)
    for n in 2:n_max
        for i in 1:(length(tokens) - n + 1)
            push!(result, join(tokens[i:i+n-1], "_"))
        end
    end
    return result
end

# ─────────────────────────────────────────────
# Vetorização TF-IDF (inferência)
# ─────────────────────────────────────────────

function vetorizar_tfidf(tokens::Vector{String},
                          vocab::Dict{String,Int},
                          idf::Vector{Float64})::Vector{Float64}
    v = zeros(Float64, length(vocab))
    total = length(tokens)
    total == 0 && return v

    freq = Dict{String,Int}()
    for t in tokens
        freq[t] = get(freq, t, 0) + 1
    end

    for (token, cnt) in freq
        if haskey(vocab, token)
            i = vocab[token]
            tf = cnt / total
            v[i] = tf * idf[i]
        end
    end

    norma = sqrt(sum(v .^ 2))
    norma > 0 && (v ./= norma)
    return v
end

# ─────────────────────────────────────────────
# Similaridade de cosseno
# ─────────────────────────────────────────────

"""
    similaridade_cosseno(a, b) -> Float64

Produto escalar entre dois vetores já normalizados (norma L2 = 1).
Resultado em [0.0, 1.0] onde 1.0 = vetores idênticos.
"""
function similaridade_cosseno(a::Vector{Float64}, b::Vector{Float64})::Float64
    return clamp(dot(a, b), 0.0, 1.0)
end

# Produto escalar simples (dot não é exportado automaticamente em todas versões)
function dot(a::Vector{Float64}, b::Vector{Float64})::Float64
    s = 0.0
    @inbounds for i in eachindex(a)
        s += a[i] * b[i]
    end
    return s
end

# ─────────────────────────────────────────────
# Carregamento do modelo treinado
# ─────────────────────────────────────────────

"""
    carregar_modelo(caminho::String) -> ModeloTreinado

Lê o arquivo de modelo gerado por treinar.jl e reconstrói as
estruturas necessárias para inferência.
"""
function carregar_modelo(caminho::String)::ModeloTreinado
    isfile(caminho) || error("""
Modelo treinado não encontrado: $caminho

Execute o treinamento primeiro:
  julia src/treinar.jl
""")

    println("Carregando modelo de: $caminho")
    dados = TOML.parsefile(caminho)

    # Metadados
    meta       = dados["meta"]
    n_vocab    = meta["n_vocab"]
    n_pergs    = meta["n_perguntas"]
    padrao     = get(meta, "padrao", "Não entendi. Pode reformular?")

    # Vocabulário
    termos = String.(dados["vocabulario"]["termos"])
    vocab  = Dict{String,Int}(t => i for (i, t) in enumerate(termos))

    # IDF
    idf = Float64.(dados["idf"]["valores"])

    # Respostas
    respostas = String.(dados["respostas"]["lista"])

    # Mapeamento pergunta → resposta
    par_idx = Int.(dados["mapeamento"]["par_idx"])

    # Vetores TF-IDF
    vets = dados["vetores"]
    vetores = Vector{Float64}[]
    for k in 1:n_pergs
        chave = "v$(k)"
        haskey(vets, chave) || error("Vetor $chave não encontrado no modelo.")
        push!(vetores, Float64.(vets[chave]))
    end

    println("✓ Modelo carregado:")
    println("  Vocabulário : $n_vocab termos")
    println("  Perguntas   : $n_pergs vetores")
    println("  Respostas   : $(length(respostas)) entradas")

    return ModeloTreinado(vocab, idf, vetores, par_idx, respostas, padrao)
end

# ─────────────────────────────────────────────
# Inferência por busca vetorial
# ─────────────────────────────────────────────

"""
    responder(modelo, entrada; limiar) -> String

Vetoriza a entrada do usuário com TF-IDF e busca o vetor de pergunta
mais próximo via similaridade de cosseno. Retorna a resposta associada
se o score estiver acima do limiar, caso contrário retorna o fallback.
"""
function responder(modelo::ModeloTreinado, entrada::String;
                   limiar::Float64=0.20)::String

    tokens = ngramas(tokenizar(entrada))
    isempty(tokens) && return modelo.padrao

    vec_entrada = vetorizar_tfidf(tokens, modelo.vocab, modelo.idf)

    # Se o vetor for nulo (todos os tokens fora do vocab), usa fallback
    all(iszero, vec_entrada) && return modelo.padrao

    melhor_score = 0.0
    melhor_k     = 0

    for (k, vec_perg) in enumerate(modelo.vetores)
        score = similaridade_cosseno(vec_entrada, vec_perg)
        if score > melhor_score
            melhor_score = score
            melhor_k     = k
        end
    end

    if melhor_score >= limiar && melhor_k > 0
        idx_resp = modelo.par_idx[melhor_k]
        return modelo.respostas[idx_resp]
    else
        return modelo.padrao
    end
end

# ─────────────────────────────────────────────
# Loop interativo
# ─────────────────────────────────────────────

"""
    iniciar(caminho_modelo::String)

Inicia o loop de conversa carregando o modelo TF-IDF pré-treinado.
"""
function iniciar(caminho_modelo::String="data/modelo_treinado.toml")
    println("=" ^ 50)
    println("  Chatbot Julia — TF-IDF + Cosseno")
    println("=" ^ 50)

    modelo = carregar_modelo(caminho_modelo)
    println("\nDigite 'sair' para encerrar.\n")

    while true
        print("Você: ")
        entrada = readline()

        entrada_norm = normalizar(entrada)

        if entrada_norm in ("sair", "exit", "quit")
            println("Bot: Até logo!")
            break
        end

        isempty(entrada_norm) && continue

        resposta = responder(modelo, entrada)
        println("Bot: $resposta\n")
    end
end

end # module chatbot

# ─────────────────────────────────────────────
# Ponto de entrada — execução direta como script
#   julia src/chatbot.jl
#   julia src/chatbot.jl data/modelo_treinado.toml
# ─────────────────────────────────────────────
if abspath(PROGRAM_FILE) == @__FILE__
    raiz_projeto  = joinpath(@__DIR__, "..")
    caminho_padrao = joinpath(raiz_projeto, "data", "modelo_treinado.toml")
    caminho_modelo = length(ARGS) >= 1 ? ARGS[1] : caminho_padrao
    chatbot.iniciar(caminho_modelo)
end


# 1. Treinar (gera data/modelo_treinado.toml)
# julia chatbot/src/treinar.jl

# 2. Rodar o chatbot (carrega os parâmetros treinados)
# julia chatbot/src/chatbot.jl