# project : autonomo
# file    : src/autonomo.jl
# desc    : rede neural recorrente autônoma de funcionamento infinito
#           — deep learning contínuo em loop, como um redemoinho que aprende enquanto gira
#
# PROMPTS
#   Ao iniciar, o sistema exibe o prompt de inicialização (banner + aquecimento + log).
#   Durante o loop, aceita dois tipos de entrada:
#     • número puro (ex: 1.5)  → estímulo direto, resposta imediata
#     • comando de texto        → interpretado pelo parse_command e despachado

using LinearAlgebra, Random, Printf, Dates
include("network.jl")
include("metrics.jl")
include("io.jl")

# ╔══════════════════════════════════════════════════════════╗
# ║                  CONFIGURAÇÃO GLOBAL                     ║
# ╚══════════════════════════════════════════════════════════╝

const CFG = (
    n_neurons        = 64,      # neurônios por camada
    n_layers         = 3,       # camadas recorrentes empilhadas
    lr               = 0.002,   # taxa de aprendizado (regra de Oja)
    oja_decay        = 0.1,     # fator de decaimento Oja (auto-normalização)
    centrifugal      = 0.97,    # dinâmica centrífuga (mantém o giro)
    noise_ambient    = 0.08,    # amplitude do ruído de fundo
    noise_burst_p    = 0.05,    # probabilidade de burst espontâneo
    noise_burst_amp  = 1.5,     # amplitude do burst
    dt               = 0.001,   # passo de tempo simulado (s)
    log_every        = 1000,    # iterações entre logs automáticos
    warmup_iters     = 1000,    # iterações de aquecimento antes do loop interativo
)

# ╔══════════════════════════════════════════════════════════╗
# ║                     INICIALIZAÇÃO                        ║
# ╚══════════════════════════════════════════════════════════╝

net         = Network(CFG.n_neurons, CFG.n_layers)
iter        = 0
sim_time    = 0.0
learn_count = 0

# ── PROMPT DE INICIALIZAÇÃO ──────────────────────────────────
# Exibe o banner, aquece a rede por warmup_iters iterações sem
# exibir cada passo, depois imprime o primeiro log de estado.
print_banner()

print("  Aquecendo a rede")
for _ in 1:CFG.warmup_iters
    global iter, sim_time, learn_count
    x = rand() < CFG.noise_burst_p ?
        randn() * CFG.noise_burst_amp :
        randn() * CFG.noise_ambient
    step_forward!(net, x, CFG.centrifugal)
    did_learn = step_learn!(net, CFG.lr, CFG.oja_decay)
    did_learn && (learn_count += 1)
    iter     += 1
    sim_time += CFG.dt
    # ponto de progresso a cada 100 iterações
    iter % 100 == 0 && print(".")
end
println(" pronto.\n")

m_init = compute_metrics(net)
print_status(m_init, sim_time, iter, learn_count)
println("  Sistema ativo. Digite um número para estimular, ou um comando (ajuda: '?').\n")

# ── CANAL DE INTERAÇÃO ASSÍNCRONA ───────────────────────────
const canal_interacao = Channel{String}(32)

@async begin
    while true
        linha = readline(stdin)
        s = strip(linha)
        isempty(s) || put!(canal_interacao, s)
    end
end

# ╔══════════════════════════════════════════════════════════╗
# ║                  PARSER DE ESTÍMULOS                     ║
# ╚══════════════════════════════════════════════════════════╝

function parse_stimulus(entrada::String, n::Int)::Union{Float64, Vector{Float64}}
    # 1. Tenta como número escalar
    v = tryparse(Float64, entrada)
    if v !== nothing
        return v
    end
    
    # 2. Tenta como array/matriz de números
    try
        expr = Meta.parse(entrada)
        val = eval(expr)
        if val isa AbstractArray && eltype(val) <: Number
            flat = Float64.(vec(val))
            if isempty(flat)
                return zeros(n)
            end
            res = zeros(n)
            for i in 1:n
                res[i] = flat[mod1(i, length(flat))]
            end
            return res
        end
    catch
    end

    # 3. Processa como texto orgânico
    bytes = Vector{UInt8}(entrada)
    if isempty(bytes)
        return zeros(n)
    end
    res = zeros(n)
    for i in 1:n
        b = bytes[mod1(i, length(bytes))]
        # Mapeia 0-255 para centralizar no zero aproximado
        res[i] = (Float64(b) - 128.0) / 64.0
    end
    return res
end

# ╔══════════════════════════════════════════════════════════╗
# ║                   LOOP INFINITO                          ║
# ╚══════════════════════════════════════════════════════════╝

while true
    global iter, sim_time, learn_count

    # ── LEITURA DE ENTRADA ───────────────────────────────────
    if isready(canal_interacao)
        entrada = take!(canal_interacao)

        comandos_conhecidos = ["?", "ajuda", "estado", "historico", "pesos", "reset", "cfg"]
        cmd_trim = lowercase(strip(entrada))

        if cmd_trim in comandos_conhecidos || startswith(cmd_trim, "burst")
            handle_command(entrada, net, CFG, sim_time, iter, learn_count)
        else
            # Processa como numérico, bloco matricial ou texto corrido
            v = parse_stimulus(entrada, CFG.n_neurons)

            # ── PROMPT DE INTERAÇÃO ──────────────────────────
            step_forward!(net, v, CFG.centrifugal)
            did_learn = step_learn!(net, CFG.lr, CFG.oja_decay)
            did_learn && (learn_count += 1)
            iter     += 1
            sim_time += CFG.dt
            m = compute_metrics(net)
            print_stimulus_response(v, m, sim_time)
        end

    else
        # Sem entrada: ruído interno, o redemoinho continua girando
        x = rand() < CFG.noise_burst_p ?
            randn() * CFG.noise_burst_amp :
            randn() * CFG.noise_ambient
        step_forward!(net, x, CFG.centrifugal)
        did_learn = step_learn!(net, CFG.lr, CFG.oja_decay)
        did_learn && (learn_count += 1)
        iter     += 1
        sim_time += CFG.dt

        if iter % CFG.log_every == 0
            m = compute_metrics(net)
            print_status(m, sim_time, iter, learn_count)
        end
    end

    sleep(0.0005)   # redemoinho gentil — não satura a CPU
end
