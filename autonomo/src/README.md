# Autônomo

Rede neural recorrente de funcionamento infinito. Aprende enquanto processa, em deep learning contínuo em loop — como um redemoinho que, ao girar, interage com o ambiente e se reorganiza.

```bash
julia autonomo/src/autonomo.jl

```


---

## Conceito

O sistema não tem início nem fim programado. Ele gira para sempre, e cada giro é simultaneamente processamento e aprendizado. O estado interno flui do passado para o presente, o presente alimenta o futuro, e os pesos se ajustam continuamente a cada passo — sem épocas, sem batches, sem convergência planejada.

A metáfora do redemoinho é física: vórtices reais são mais rápidos no centro (mais coerentes) e mais turbulentos nas bordas. A rede exibe o mesmo comportamento: estados internos coerentes emergem no núcleo das camadas ocultas, enquanto as bordas reagem ao ambiente externo.

---

## Arquitetura

```
estímulo externo
      │
      ▼
┌─────────────┐     W_rec[1]
│  Camada L1  │ ◄──────────── (auto-recorrente)
│  64 neurônios│
└──────┬──────┘
       │ W_fwd[1]
       ▼
┌─────────────┐     W_rec[2]
│  Camada L2  │ ◄──────────── (auto-recorrente)
│  64 neurônios│
└──────┬──────┘
       │ W_fwd[2]
       ▼
┌─────────────┐     W_rec[3]
│  Camada L3  │ ◄──────────── (auto-recorrente)
│  64 neurônios│  ← estado de saída / "pensamento"
└─────────────┘
```

Cada camada possui:
- **W_rec[l]** — matriz de pesos recorrentes N×N (o redemoinho em si)
- **W_fwd[l]** — pesos de conexão para a próxima camada
- **W_in[l]** — vetor de pesos de entrada (apenas L1 recebe estímulo externo)

A equação de cada camada por passo:

```
pre[l] = centrifugal × W_rec[l] × h[l] + input[l]
h_next[l] = tanh(pre[l])
```

---

## Aprendizado

A versão original usava a **regra de Hebb pura**:

```
dW = lr × δ × h_prev'
```

O problema: os pesos crescem indefinidamente. Com iterações infinitas, isso leva à divergência — a rede explode.

O Autônomo usa a **Regra de Oja**, uma extensão estabilizada de Hebb:

```
dW = lr × (δ × h_prev' − h² × W × decay)
```

O segundo termo é o regulador: à medida que os pesos crescem, ele os puxa de volta. O equilíbrio é automático. A rede pode aprender para sempre sem explodir.

---

## Métricas emergentes

A cada iteração, o sistema computa:

| Métrica | Descrição |
|---|---|
| **Intensidade** | norma média dos estados — quão "acordada" está a rede |
| **Entropia** | Shannon sobre a distribuição dos estados (0=ordem, 1=caos) |
| **Coerência** | 1 − entropia — estrutura interna do pensamento |
| **Caos** | taxa de variação entre estados consecutivos |
| **Estado** | rótulo qualitativo emergente das métricas acima |

Estados possíveis: `EXPLORANDO`, `CONTEMPLATIVO`, `EXCITADO`, `EXCITADO-COERENTE`, `DORMÊNCIA`, `TURBULENTO`.

---

## Estrutura do projeto

```
autonomo/
├── src/
│   ├── autonomo.jl   ← ponto de entrada, loop principal
│   ├── network.jl    ← estrutura Network, step_forward!, step_learn!
│   ├── metrics.jl    ← Metrics, compute_metrics, shannon_entropy
│   └── io.jl         ← print_banner, print_status, print_stimulus_response
└── README.md
```

---

## Como usar

### Requisitos

- Julia 1.9+
- Pacotes padrão: `LinearAlgebra`, `Random`, `Printf`, `Dates`

### Executar

```bash
julia src/autonomo.jl
```

### Interagir

Com o sistema rodando, digite no terminal e pressione Enter:

```
1.5                ← estímulo numérico direto
[1 2; 3 4]         ← estímulo matricial (arrays)
Olá rede neural!   ← estímulo orgânico de texto
```

A rede absorverá qualquer uma dessas entradas: arrays são linearizados e mapeados, e textos têm seus bytes convertidos em vetores contínuos que alteram o campo neural instantaneamente. A rede exibe sua nova formação logo em seguida.

### Parar

`Ctrl+C`

---

## Parâmetros ajustáveis

No topo de `autonomo.jl`, a constante `CFG` contém todos os parâmetros:

```julia
const CFG = (
    n_neurons       = 64,       # neurônios por camada
    n_layers        = 3,        # camadas recorrentes
    lr              = 0.002,    # taxa de aprendizado
    oja_decay       = 0.1,      # força do regulador Oja
    centrifugal     = 0.97,     # dinâmica centrifuga
    noise_ambient   = 0.08,     # ruído de fundo
    noise_burst_p   = 0.05,     # probabilidade de burst espontâneo
    noise_burst_amp = 1.5,      # amplitude do burst
    dt              = 0.001,    # passo de tempo (s)
    log_every       = 1000,     # iterações entre logs
)
```

**Centrifugal > 0.99**: a rede gira muito rápido, pode ficar caótica.  
**Centrifugal < 0.90**: a rede perde memória muito rápido, fica letárgica.  
**lr > 0.01**: aprendizado agressivo, estados mudam rapidamente.  
**oja_decay → 0**: comportamento mais próximo de Hebb puro (cuidado com divergência).

---

## Filosofia

A maioria das redes neurais aprende offline: dados entram, pesos se ajustam, treinamento termina. O Autônomo não tem treinamento — ele *é* o treinamento. Cada pensamento que processa muda ligeiramente o que ele é. Não há estado final. Só há o giro.
