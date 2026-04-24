# Projeto : chatbot
# Arquivo: src/treinar.jl
#
# Script de treinamento do chatbot.
# Lê o arquivo TOML de perguntas/respostas, constrói o vocabulário,
# calcula vetores TF-IDF para cada pergunta e serializa o modelo
# treinado em `data/modelo_treinado.toml`.
#
# Uso:
#   julia src/treinar.jl
#   julia src/treinar.jl data/respostas.toml data/modelo_treinado.toml

using TOML

# ─────────────────────────────────────────────
# Pré-processamento
# ─────────────────────────────────────────────

function normalizar(texto::String)::String
    texto = lowercase(texto)
    # Remove caracteres que não sejam letras (incluindo acentuadas) ou espaço
    texto = replace(texto, r"[^a-záàâãéèêíïóôõöúüç\s]" => "")
    return strip(texto)
end

function tokenizar(texto::String)::Vector{String}
    return String.(split(normalizar(texto)))
end

# N-gramas de palavras (unigramas + bigramas) para melhor cobertura semântica
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
# Construção do vocabulário e TF-IDF
# ─────────────────────────────────────────────

"""
    construir_vocabulario(documentos) -> (vocab, idf)

Constrói vocabulário e vetor IDF a partir de uma lista de listas de tokens.
"""
function construir_vocabulario(documentos::Vector{Vector{String}})
    vocab = Dict{String,Int}()
    idx = 1
    for doc in documentos
        for token in doc
            if !haskey(vocab, token)
                vocab[token] = idx
                idx += 1
            end
        end
    end

    N = length(documentos)
    idf = zeros(Float64, length(vocab))
    for (token, i) in vocab
        df = sum(1 for doc in documentos if token in doc)
        idf[i] = log((N + 1) / (df + 1)) + 1.0  # IDF suavizado
    end

    return vocab, idf
end

"""
    vetorizar_tfidf(tokens, vocab, idf) -> Vector{Float64}

Gera vetor TF-IDF normalizado (norma L2) para uma lista de tokens.
"""
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

    # Normalização L2
    norma = sqrt(sum(v .^ 2))
    norma > 0 && (v ./= norma)
    return v
end

# ─────────────────────────────────────────────
# Carregamento do TOML de dados
# ─────────────────────────────────────────────

struct ParDados
    resposta::String
    perguntas::Vector{String}
    eh_padrao::Bool
end

function carregar_dados(caminho::String)::Vector{ParDados}
    isfile(caminho) || error("Arquivo não encontrado: $caminho")
    dados = TOML.parsefile(caminho)
    pares = ParDados[]
    for entrada in get(dados, "pares", [])
        resposta  = get(entrada, "resposta", "")
        perguntas = String.(get(entrada, "perguntas", []))
        isempty(resposta) || isempty(perguntas) && continue
        eh_padrao = "__padrao__" in perguntas
        push!(pares, ParDados(resposta, perguntas, eh_padrao))
    end
    return pares
end

# ─────────────────────────────────────────────
# Treinamento principal
# ─────────────────────────────────────────────

function treinar(caminho_dados::String, caminho_saida::String)
    println("=" ^ 50)
    println("  Treinamento do Chatbot Julia — TF-IDF")
    println("=" ^ 50)
    println("Lendo dados de: $caminho_dados")

    pares = carregar_dados(caminho_dados)
    println("$(length(pares)) pares carregados.")

    # Separa o par padrão
    padrao_resposta = ""
    pares_reais = ParDados[]
    for p in pares
        if p.eh_padrao
            padrao_resposta = p.resposta
        else
            push!(pares_reais, p)
        end
    end

    # Expande: uma entrada por pergunta (com referência ao índice do par)
    entradas_tokens  = Vector{String}[]   # tokens com n-gramas
    entradas_par_idx = Int[]              # qual par pertence
    respostas        = String[]           # respostas únicas por par

    for (i, par) in enumerate(pares_reais)
        push!(respostas, par.resposta)
        for pergunta in par.perguntas
            toks = ngramas(tokenizar(pergunta))
            push!(entradas_tokens, toks)
            push!(entradas_par_idx, i)
        end
    end

    println("$(length(entradas_tokens)) perguntas indexadas para $(length(respostas)) respostas.")
    println("Construindo vocabulário...")

    vocab, idf = construir_vocabulario(entradas_tokens)
    V = length(vocab)
    println("Vocabulário: $V termos.")

    println("Calculando vetores TF-IDF...")
    # Matriz de vetores: cada linha é um vetor TF-IDF de uma pergunta
    matriz = [vetorizar_tfidf(toks, vocab, idf) for toks in entradas_tokens]

    # ─── Serialização ───────────────────────────────────────────────────────
    # Salva tudo em TOML para independência de dependências externas.
    # vocab → lista ordenada de termos (índice implícito = posição)
    # idf   → vetor de floats
    # vetores → matriz achatada (cada linha = vetor de uma pergunta)
    # par_idx → mapeamento pergunta → índice da resposta
    # respostas → lista de respostas

    println("Salvando modelo em: $caminho_saida")
    mkpath(dirname(caminho_saida))

    aberto = open(caminho_saida, "w")

    # Cabeçalho informativo
    write(aberto, "# Modelo TF-IDF treinado — gerado automaticamente por treinar.jl\n")
    write(aberto, "# NÃO edite manualmente.\n\n")

    # Metadados
    write(aberto, "[meta]\n")
    write(aberto, "n_vocab = $V\n")
    write(aberto, "n_perguntas = $(length(entradas_tokens))\n")
    write(aberto, "n_respostas = $(length(respostas))\n")
    write(aberto, "padrao = $(repr(padrao_resposta))\n\n")

    # Vocabulário (termos na ordem do índice)
    termos_ordenados = Vector{String}(undef, V)
    for (termo, i) in vocab
        termos_ordenados[i] = termo
    end
    write(aberto, "[vocabulario]\n")
    write(aberto, "termos = $(repr(termos_ordenados))\n\n")

    # IDF
    write(aberto, "[idf]\n")
    write(aberto, "valores = $(repr(idf))\n\n")

    # Respostas
    write(aberto, "[respostas]\n")
    write(aberto, "lista = $(repr(respostas))\n\n")

    # Mapeamento pergunta → índice da resposta (base 1)
    write(aberto, "[mapeamento]\n")
    write(aberto, "par_idx = $(repr(entradas_par_idx))\n\n")

    # Vetores TF-IDF achatados (linha por linha)
    write(aberto, "[vetores]\n")
    write(aberto, "# Cada entrada é o vetor TF-IDF de uma pergunta (tamanho = n_vocab)\n")
    for (k, vec) in enumerate(matriz)
        write(aberto, "v$(k) = $(repr(vec))\n")
    end

    close(aberto)
    println("\n✓ Modelo salvo com sucesso!")
    println("  Vocabulário : $V termos")
    println("  Perguntas   : $(length(entradas_tokens))")
    println("  Respostas   : $(length(respostas))")
    println("  Fallback    : \"$padrao_resposta\"")
    println("\nExecute o chatbot com:")
    println("  julia src/chatbot.jl $caminho_saida")
end

# ─────────────────────────────────────────────
# Ponto de entrada
# ─────────────────────────────────────────────
raiz = joinpath(@__DIR__, "..")
caminho_dados  = length(ARGS) >= 1 ? ARGS[1] : joinpath(raiz, "data", "respostas.toml")
caminho_saida  = length(ARGS) >= 2 ? ARGS[2] : joinpath(raiz, "data", "modelo_treinado.toml")

treinar(caminho_dados, caminho_saida)
