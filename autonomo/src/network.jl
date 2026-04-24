# Projeto : autonomo
# Arquivo: src/network.jl
# Desc    : estrutura da rede neural recorrente (RNN) com N camadas empilhadas
#           cada camada tem pesos recorrentes próprios; camadas se alimentam em cascata

# ╔══════════════════════════════════════════════════════════╗
# ║                    ESTRUTURA DA REDE                     ║
# ╚══════════════════════════════════════════════════════════╝

mutable struct Network
    n          :: Int                        # neurônios por camada
    n_layers   :: Int                        # número de camadas

    W_rec      :: Vector{Matrix{Float64}}    # pesos recorrentes  [L][N×N]
    W_fwd      :: Vector{Matrix{Float64}}    # pesos forward      [L-1][N×N]
    W_in       :: Vector{Vector{Float64}}    # pesos de entrada   [L][N]

    h          :: Vector{Vector{Float64}}    # estado atual       [L][N]
    h_prev     :: Vector{Vector{Float64}}    # estado anterior    [L][N]

    # histórico de normas por camada — usado pelo prompt de interação
    # para descrever a trajetória do estado entre estímulos
    norm_history :: Vector{Vector{Float64}}  # [L][últimas K normas]
end

const HISTORY_LEN = 8   # quantos passos de norma são mantidos por camada

"""
    Network(n, n_layers)

Constrói uma rede recorrente com `n` neurônios e `n_layers` camadas.

Pesos recorrentes são inicializados próximos da identidade com ruído gaussiano
pequeno — garante que o "giro" inicial seja estável antes do aprendizado.
"""
function Network(n::Int, n_layers::Int)
    rng = MersenneTwister()

    W_rec        = Vector{Matrix{Float64}}(undef, n_layers)
    W_fwd        = Vector{Matrix{Float64}}(undef, n_layers - 1)
    W_in         = Vector{Vector{Float64}}(undef, n_layers)
    h            = Vector{Vector{Float64}}(undef, n_layers)
    h_prev       = Vector{Vector{Float64}}(undef, n_layers)
    norm_history = [zeros(HISTORY_LEN) for _ in 1:n_layers]

    for l in 1:n_layers
        # Próximo de identidade + ruído pequeno → dinâmica estável inicial
        W_rec[l] = Matrix{Float64}(I, n, n) * 0.9 .+ randn(rng, n, n) .* 0.05
        W_in[l]  = randn(rng, n)
        h[l]     = randn(rng, n) .* 0.1
        h_prev[l]= zeros(n)
    end

    for l in 1:(n_layers - 1)
        # Conexões forward: inicialização Xavier
        scale    = sqrt(2.0 / (n + n))
        W_fwd[l] = randn(rng, n, n) .* scale
    end

    Network(n, n_layers, W_rec, W_fwd, W_in, h, h_prev, norm_history)
end

# ╔══════════════════════════════════════════════════════════╗
# ║                   PASSO FORWARD                          ║
# ╚══════════════════════════════════════════════════════════╝

"""
    step_forward!(net, stimulus, centrifugal)

Processa um passo temporal em todas as camadas.

A camada 1 recebe o estímulo externo via W_in.
As camadas 2..L recebem a saída da camada anterior via W_fwd.
Cada camada se realimenta do próprio estado via W_rec.

Equação por camada:
    pre        = centrifugal × W_rec[l] × h[l] + input[l]
    h_next[l]  = tanh(pre)

Atualiza norm_history para rastreamento de trajetória.
"""
function step_forward!(net::Network, stimulus::Union{Float64, Vector{Float64}}, centrifugal::Float64)
    n = net.n

    for l in 1:net.n_layers
        net.h_prev[l] .= net.h[l]

        rec = net.W_rec[l] * net.h[l]

        inp = if l == 1
            net.W_in[l] .* stimulus
        else
            net.W_fwd[l - 1] * net.h[l - 1]
        end

        pre         = centrifugal .* rec .+ inp
        net.h[l]   .= tanh.(pre)

        # desloca o histórico e registra norma atual
        net.norm_history[l][1:end-1] .= net.norm_history[l][2:end]
        net.norm_history[l][end]      = norm(net.h[l]) / sqrt(n)
    end
end

# ╔══════════════════════════════════════════════════════════╗
# ║               APRENDIZADO (REGRA DE OJA)                 ║
# ╚══════════════════════════════════════════════════════════╝

"""
    step_learn!(net, lr, decay)

Aplica a regra de Oja em cada camada recorrente.

Hebb puro: dW = lr × δ × h_prev'        →  diverge com o tempo
Oja:       dW = lr × (δ × h_prev' − h² × W × decay)
           o segundo termo auto-normaliza, mantendo ‖W‖ estável
           para sempre — permite aprendizado infinito sem explosão.

Retorna `true` se houve mudança significativa de estado (aprendizado real).
"""
function step_learn!(net::Network, lr::Float64, decay::Float64)::Bool
    learned = false

    for l in 1:net.n_layers
        δ = net.h[l] .- net.h_prev[l]

        if norm(δ) > 1e-4
            dW = lr .* (δ * net.h_prev[l]' .- (net.h[l] .^ 2) .* net.W_rec[l] .* decay)
            net.W_rec[l] .+= dW
            learned = true
        end
    end

    return learned
end

# ╔══════════════════════════════════════════════════════════╗
# ║              CONSULTA DE ESTADO INTERNO                  ║
# ╚══════════════════════════════════════════════════════════╝

"""
    weight_stats(net)

Retorna (norma_media, norma_max) dos pesos recorrentes de todas as camadas.
Usado pelo comando 'pesos' do prompt de interação.
"""
function weight_stats(net::Network)
    norms = [norm(net.W_rec[l]) for l in 1:net.n_layers]
    return mean_f(norms), maximum(norms)
end

mean_f(v) = sum(v) / length(v)
