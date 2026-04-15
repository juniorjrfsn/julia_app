# project : autonomo
# file    : src/io.jl
# desc    : formatação de saída no terminal + parser de comandos de texto
#
# PROMPTS
#   print_banner()              → prompt de inicialização (cabeçalho)
#   print_status()              → log periódico automático
#   print_stimulus_response()   → prompt de interação (resposta a estímulo numérico)
#   handle_command()            → interpreta comandos de texto digitados pelo usuário

# ╔══════════════════════════════════════════════════════════╗
# ║               PROMPT DE INICIALIZAÇÃO                    ║
# ╚══════════════════════════════════════════════════════════╝

"""
    print_banner()

Exibe o cabeçalho do sistema ao iniciar (prompt de inicialização).
"""
function print_banner()
    println()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║           A U T Ô N O M O  —  rede neural infinita          ║")
    println("║    deep learning contínuo · redemoinho que aprende ao girar  ║")
    println("╠══════════════════════════════════════════════════════════════╣")
    @printf("║  Arquitetura: RNN %d camadas × %d neurônios%s║\n",
            CFG.n_layers, CFG.n_neurons,
            " " ^ (29 - ndigits(CFG.n_layers) - ndigits(CFG.n_neurons)))
    println("║  Aprendizado: Regra de Oja (auto-normalizada, estável ∞)     ║")
    println("║  Interação:   número → estímulo direto  |  '?' → ajuda       ║")
    println("║  Parar:       Ctrl+C                                          ║")
    println("╚══════════════════════════════════════════════════════════════╝")
    println()
end

# ╔══════════════════════════════════════════════════════════╗
# ║                    LOG PERIÓDICO                         ║
# ╚══════════════════════════════════════════════════════════╝

"""
    print_status(m, sim_time, iter, learn_count)

Log periódico automático do estado da rede.
"""
function print_status(m::Metrics, sim_time::Float64, iter::Int, learn_count::Int)
    bar_i = progress_bar(m.intensity, 20)
    bar_c = progress_bar(m.coherence, 20)
    bar_x = progress_bar(m.chaos,     20)

    @printf("┌─ t=%.2fs  iter=%-8d  aprendizados=%-6d\n",
            sim_time, iter, learn_count)
    @printf("│  intensidade  %s  %.3f\n", bar_i, m.intensity)
    @printf("│  coerência    %s  %.3f\n", bar_c, m.coherence)
    @printf("│  caos         %s  %.3f  entropia=%.3f\n", bar_x, m.chaos, m.entropy)
    @printf("│  camadas      %s\n",
            join([@sprintf("L%d=%.3f", i, v) for (i,v) in enumerate(m.layer_norms)], "  "))
    @printf("└─ estado: %s\n\n", m.mood)
end

# ╔══════════════════════════════════════════════════════════╗
# ║              PROMPT DE INTERAÇÃO — ESTÍMULO              ║
# ╚══════════════════════════════════════════════════════════╝

"""
    print_stimulus_response(stimulus, m, sim_time)

Exibe a resposta da rede a um estímulo numérico externo.
Inclui nota sobre transição de estado quando ela ocorre.
"""
function print_stimulus_response(stimulus::Union{Float64, Vector{Float64}}, m::Metrics, sim_time::Float64)
    println()
    println("╔═══════════════════════════════════════╗")
    if stimulus isa Float64
        @printf("║  ↯ ESTÍMULO EXTERNO: %+8.4f         ║\n", stimulus)
    else
        @printf("║  ↯ ESTÍMULO EXTERNO: [%-14s]║\n", "TEXTO/MATRIZ")
    end
    println("╠═══════════════════════════════════════╣")
    @printf("║  t=%.2fs                              ║\n", sim_time)
    @printf("║  Intensidade:  %.4f                ║\n", m.intensity)
    @printf("║  Coerência:    %.4f                ║\n", m.coherence)
    @printf("║  Caos:         %.4f                ║\n", m.chaos)
    @printf("║  Entropia:     %.4f                ║\n", m.entropy)
    @printf("║  Estado:       %-20s ║\n", m.mood)

    if m.delta_mood
        @printf("║  Transição:    %-8s → %-8s   ║\n", m.prev_mood, m.mood)
    end

    println("╚═══════════════════════════════════════╝")

    # Observação qualitativa baseada na magnitude e sinal do estímulo
    println(stimulus_observation(stimulus, m))
    println()
end

"""
    stimulus_observation(stimulus, m)

Gera uma linha de observação qualitativa sobre como o redemoinho
reagiu ao estímulo — baseada no sinal, magnitude e estado resultante.
"""
function stimulus_observation(stimulus::Union{Float64, Vector{Float64}}, m::Metrics)::String
    if stimulus isa Float64
        mag = abs(stimulus)
        sinal = stimulus > 0 ? "positivo" : "negativo"
    else
        mag = norm(stimulus)
        sinal = sum(stimulus) >= 0 ? "positivo" : "negativo"
    end

    if mag < 0.1
        base = "Perturbação sutil — o giro mal percebeu a entrada."
    elseif mag < 1.0
        base = "Estímulo $sinal moderado — o redemoinho reajustou seu eixo."
    elseif mag < 3.0
        base = "Entrada $sinal forte — ondas visíveis nas camadas internas."
    else
        base = "Impulso $sinal intenso — o vórtice foi chacoalhado até o núcleo."
    end

    sufixo = if m.mood == "TURBULENTO"
        " O sistema perdeu coerência transitoriamente."
    elseif m.mood == "EXCITADO-COERENTE"
        " Energia alta com estrutura preservada — estado ótimo."
    elseif m.mood == "CONTEMPLATIVO"
        " Após o impacto, a rede se reorganizou em silêncio."
    elseif m.mood == "DORMÊNCIA"
        " O estímulo não foi suficiente para despertar o sistema."
    else
        ""
    end

    return "  ↳ $base$sufixo"
end

# ╔══════════════════════════════════════════════════════════╗
# ║           PROMPT DE INTERAÇÃO — COMANDOS DE TEXTO        ║
# ╚══════════════════════════════════════════════════════════╝

"""
    handle_command(entrada, net, cfg, sim_time, iter, learn_count)

Interpreta comandos de texto digitados pelo usuário durante o loop.

Comandos disponíveis:
  ?  / ajuda      — lista todos os comandos
  estado          — exibe métricas detalhadas do momento atual
  historico       — mostra trajetória recente de normas por camada
  pesos           — estatísticas dos pesos recorrentes (norma média e máxima)
  reset           — reinicia o estado interno h para zeros (pesos são mantidos)
  burst <amp>     — injeta manualmente um burst de amplitude <amp>
  cfg             — exibe a configuração atual (CFG)
"""
function handle_command(entrada::String,
                        net::Network,
                        cfg,
                        sim_time::Float64,
                        iter::Int,
                        learn_count::Int)

    cmd = lowercase(strip(entrada))

    if cmd == "?" || cmd == "ajuda"
        print_help()

    elseif cmd == "estado"
        m = compute_metrics(net)
        println("\n── Estado atual ──────────────────────────────")
        print_status(m, sim_time, iter, learn_count)

    elseif cmd == "historico"
        println("\n── Histórico de normas (últimos $(HISTORY_LEN) passos) ──")
        for l in 1:net.n_layers
            vals = [@sprintf("%.3f", v) for v in net.norm_history[l]]
            println("  L$l: $(join(vals, "  "))")
        end
        println()

    elseif cmd == "pesos"
        media, maximo = weight_stats(net)
        @printf("\n── Pesos recorrentes ──────────────────────────\n")
        @printf("  Norma média: %.4f\n", media)
        @printf("  Norma máx:   %.4f\n\n", maximo)

    elseif cmd == "reset"
        for l in 1:net.n_layers
            fill!(net.h[l],      0.0)
            fill!(net.h_prev[l], 0.0)
        end
        println("\n  [RESET] Estado interno zerado. Pesos preservados.\n")

    elseif startswith(cmd, "burst")
        parts = split(cmd)
        amp = length(parts) >= 2 ? something(tryparse(Float64, parts[2]), cfg.noise_burst_amp) :
                                   cfg.noise_burst_amp
        v = randn() * amp
        step_forward!(net, v, cfg.centrifugal)
        step_learn!(net, cfg.lr, cfg.oja_decay)
        m = compute_metrics(net)
        @printf("\n  [BURST] amplitude=%.2f  estímulo gerado=%+.4f\n", amp, v)
        print_stimulus_response(v, m, sim_time)

    elseif cmd == "cfg"
        println("\n── Configuração atual ────────────────────────")
        for f in fieldnames(typeof(cfg))
            @printf("  %-18s = %s\n", f, getfield(cfg, f))
        end
        println()

    else
        println("\n  [AVISO] Comando desconhecido: \"$entrada\"")
        println("  Digite '?' para ver os comandos disponíveis.\n")
    end
end

"""
    print_help()

Exibe a lista de comandos interativos disponíveis.
"""
function print_help()
    println()
    println("╔══════════════════════════════════════════════════════════╗")
    println("║                    COMANDOS DISPONÍVEIS                  ║")
    println("╠══════════════════════════════════════════════════════════╣")
    println("║  <número>         estímulo direto (ex: 1.5  -3.0  0.01) ║")
    println("║  estado           métricas detalhadas do momento atual   ║")
    println("║  historico        trajetória de normas por camada        ║")
    println("║  pesos            estatísticas dos pesos recorrentes     ║")
    println("║  burst [amp]      injeta burst manual (padrão: 1.5)     ║")
    println("║  reset            zera o estado interno (pesos mantidos) ║")
    println("║  cfg              exibe todos os parâmetros do CFG       ║")
    println("║  ?  /  ajuda      exibe esta mensagem                    ║")
    println("║  Ctrl+C           encerra o sistema                      ║")
    println("╚══════════════════════════════════════════════════════════╝")
    println()
end

# ╔══════════════════════════════════════════════════════════╗
# ║                    UTILITÁRIOS                           ║
# ╚══════════════════════════════════════════════════════════╝

"""
    progress_bar(value, width)

Gera uma barra de progresso ASCII para um valor em [0, 1].
"""
function progress_bar(value::Float64, width::Int)::String
    filled = clamp(round(Int, value * width), 0, width)
    "█" ^ filled * "░" ^ (width - filled)
end
