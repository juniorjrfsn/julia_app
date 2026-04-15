# project : autonomo
# file    : src/metrics.jl
# desc    : métricas emergentes do estado interno da rede
#           — intensidade, entropia, coerência, caos, estado emocional

# ╔══════════════════════════════════════════════════════════╗
# ║                    ESTRUTURA DE MÉTRICAS                  ║
# ╚══════════════════════════════════════════════════════════╝

struct Metrics
    intensity    :: Float64          # norma média do estado (quão "acordada" a rede está)
    entropy      :: Float64          # entropia de Shannon dos estados (0=ordem, 1=caos)
    coherence    :: Float64          # 1 − entropy (coerência interna)
    chaos        :: Float64          # taxa de variação entre estados consecutivos
    layer_norms  :: Vector{Float64}  # norma de cada camada individualmente
    mood         :: String           # estado emergente qualitativo
    delta_mood   :: Bool             # true se o mood mudou em relação ao passo anterior
    prev_mood    :: String           # mood do passo anterior (para o prompt de interação)
end

# ── estado global do mood anterior ──────────────────────────
# Mantido aqui para que compute_metrics possa detectar transições
# sem precisar receber o mood anterior como argumento.
const _last_mood = Ref("EXPLORANDO")

# ╔══════════════════════════════════════════════════════════╗
# ║                   FUNÇÕES DE CÁLCULO                     ║
# ╚══════════════════════════════════════════════════════════╝

"""
    shannon_entropy(v)

Entropia de Shannon normalizada para um vetor de ativações.
Usa histograma de 8 bins no intervalo [-1, 1].
Retorna valor em [0, 1] onde 1 é máxima desordem.
"""
function shannon_entropy(v::Vector{Float64})::Float64
    n_bins = 8
    counts = zeros(Int, n_bins)
    for x in v
        bin = clamp(floor(Int, (x + 1.0) / 2.0 * n_bins) + 1, 1, n_bins)
        counts[bin] += 1
    end
    total = length(v)
    e = 0.0
    for c in counts
        p = c / total
        if p > 0
            e -= p * log2(p)
        end
    end
    return e / log2(n_bins)
end

"""
    compute_metrics(net)

Calcula todas as métricas emergentes da rede no passo atual.
Detecta transição de mood e armazena em delta_mood / prev_mood.
"""
function compute_metrics(net::Network)::Metrics
    n = net.n

    layer_norms = [norm(net.h[l]) / sqrt(n) for l in 1:net.n_layers]
    intensity   = mean_f(layer_norms)

    entropy   = shannon_entropy(net.h[net.n_layers])
    coherence = 1.0 - entropy

    delta_norms = [norm(net.h[l] .- net.h_prev[l]) / sqrt(n) for l in 1:net.n_layers]
    chaos = clamp(mean_f(delta_norms) * 4.0, 0.0, 1.0)

    mood       = determine_mood(intensity, coherence, chaos)
    prev       = _last_mood[]
    transicao  = (mood != prev)
    _last_mood[] = mood

    Metrics(intensity, entropy, coherence, chaos, layer_norms, mood, transicao, prev)
end

"""
    determine_mood(intensity, coherence, chaos)

Determina o estado emergente qualitativo da rede com base nas métricas.

Hierarquia de prioridade (do mais urgente ao mais suave):
  1. TURBULENTO      — caos dominante
  2. EXCITADO-COERENTE — alta energia + estrutura interna
  3. EXCITADO        — alta energia sem estrutura
  4. CONTEMPLATIVO   — baixo caos, alta coerência
  5. DORMÊNCIA       — energia mínima
  6. EXPLORANDO      — estado neutro de fundo
"""
function determine_mood(intensity::Float64, coherence::Float64, chaos::Float64)::String
    chaos > 0.65                          && return "TURBULENTO"
    intensity > 0.7 && coherence > 0.55   && return "EXCITADO-COERENTE"
    intensity > 0.7                        && return "EXCITADO"
    coherence > 0.65                       && return "CONTEMPLATIVO"
    intensity < 0.15                       && return "DORMÊNCIA"
    return "EXPLORANDO"
end

# Helper — evita conflito com mean de Statistics
mean_f(v) = sum(v) / length(v)
