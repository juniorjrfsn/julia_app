# 🎥 CNNCheckin - Sistema de Reconhecimento Facial com Webcam

Sistema completo de reconhecimento facial usando Deep Learning (CNN) com Julia, com suporte total para captura via webcam.

## 📋 Índice

- [Características](#características)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Guia Rápido](#guia-rápido)
- [Uso Detalhado](#uso-detalhado)
- [Casos de Uso](#casos-de-uso)
- [Troubleshooting](#troubleshooting)

## ✨ Características

### 🎯 Funcionalidades Principais

- **Captura via Webcam**: Captura imagens diretamente da câmera
- **Treinamento Inicial**: Treina modelo CNN com múltiplas pessoas
- **Aprendizado Incremental**: Adiciona novas pessoas sem retreinar tudo
- **Identificação em Tempo Real**: Identifica pessoas instantaneamente
- **Autenticação**: Verifica identidade de pessoa específica
- **Modo Contínuo**: Monitoramento constante
- **Sistema Check-in/Check-out**: Registro de presença automático

### 🔧 Tecnologias

- **Julia 1.9+**: Linguagem de programação de alta performance
- **Flux.jl**: Deep Learning framework
- **VideoIO.jl**: Captura de vídeo e webcam
- **Images.jl**: Processamento de imagens
- **CNN**: Rede Neural Convolucional customizada

## 📦 Requisitos

### Sistema Operacional
- Linux, macOS ou Windows
- Webcam conectada

### Software
```julia
# Julia 1.9 ou superior
using Pkg

# Pacotes necessários
Pkg.add([
    "Flux",
    "Images",
    "FileIO",
    "CUDA",  # Opcional, para GPU
    "Statistics",
    "Random",
    "JLD2",
    "TOML",
    "ImageTransformations",
    "LinearAlgebra",
    "Dates",
    "ImageView",
    "VideoIO"
])
```

### Hardware
- **Mínimo**: CPU 2+ cores, 4GB RAM, Webcam
- **Recomendado**: CPU 4+ cores, 8GB RAM, GPU NVIDIA (opcional)

## 🚀 Instalação

### 1. Instalar Julia

```bash
# Linux/macOS
wget https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.4-linux-x86_64.tar.gz
tar -xvzf julia-1.9.4-linux-x86_64.tar.gz
sudo mv julia-1.9.4 /opt/
sudo ln -s /opt/julia-1.9.4/bin/julia /usr/local/bin/julia

# Ou use o gerenciador de pacotes do seu sistema
```

### 2. Clonar/Criar Estrutura do Projeto

```bash
mkdir -p cnncheckin/{src,dados/{fotos_train,fotos_new,fotos_auth}}
cd cnncheckin
```

### 3. Copiar Arquivos do Sistema

Coloque os seguintes arquivos em `cnncheckin/src/`:
- `cnncheckin_core.jl`
- `cnncheckin_webcam.jl`
- `cnncheckin_capture.jl`
- `cnncheckin_pretrain_webcam.jl`
- `cnncheckin_incremental_webcam.jl`
- `cnncheckin_identify_webcam.jl`

### 4. Instalar Dependências

```bash
cd src
julia --project -e 'using Pkg; Pkg.add(["Flux", "Images", "FileIO", "VideoIO", "ImageView", "JLD2", "TOML", "ImageTransformations"])'
```

### 5. Testar Câmera

```bash
julia cnncheckin_capture.jl --cameras
```

## 📁 Estrutura do Projeto

```
cnncheckin/
├── src/
│   ├── cnncheckin_core.jl                    # Módulo central
│   ├── cnncheckin_webcam.jl                  # Módulo de webcam
│   ├── cnncheckin_capture.jl                 # Interface de captura
│   ├── cnncheckin_pretrain_webcam.jl         # Treinamento inicial
│   ├── cnncheckin_incremental_webcam.jl      # Aprendizado incremental
│   ├── cnncheckin_identify_webcam.jl         # Identificação
│   ├── face_recognition_model.jld2           # Modelo treinado (gerado)
│   ├── face_recognition_config.toml          # Configuração (gerado)
│   └── face_recognition_model_data.toml      # Metadados (gerado)
├── dados/
│   ├── fotos_train/                          # Imagens de treinamento
│   ├── fotos_new/                            # Imagens incrementais
│   └── fotos_auth/                           # Imagens de autenticação
└── README_WEBCAM.md                          # Este arquivo
```

## ⚡ Guia Rápido

### Fluxo Completo em 3 Passos

#### 1️⃣ **Capturar e Treinar (Primeiras Pessoas)**

```bash
cd src

# Modo rápido: capturar 3 pessoas com 15 fotos cada
julia cnncheckin_pretrain_webcam.jl --quick "João Silva" "Maria Santos" "Pedro Costa" --num 15
```

O sistema irá:
- Capturar 15 fotos de cada pessoa sequencialmente
- Treinar o modelo CNN automaticamente
- Salvar modelo e configurações

**Tempo estimado**: 10-15 minutos

#### 2️⃣ **Adicionar Novas Pessoas**

```bash
# Adicionar 2 novas pessoas com 10 fotos cada
julia cnncheckin_incremental_webcam.jl --quick "Carlos Alberto" "Ana Paula" --num 10
```

O sistema irá:
- Capturar fotos das novas pessoas
- Treinar incrementalmente (sem retreinar tudo)
- Atualizar o modelo

**Tempo estimado**: 5-8 minutos

#### 3️⃣ **Identificar Pessoas**

```bash
# Modo interativo (recomendado)
julia cnncheckin_identify_webcam.jl

# Ou identificação direta
julia cnncheckin_identify_webcam.jl --identify
```

## 📖 Uso Detalhado

### 1. Captura de Imagens

#### Modo Interativo
```bash
julia cnncheckin_capture.jl
```

Menu com opções:
1. Capturar para treinamento inicial
2. Capturar para aprendizado incremental
3. Capturar para identificação
4. Testar câmera
5. Listar câmeras

#### Captura para Treinamento
```bash
# Capturar 15 fotos de uma pessoa
julia cnncheckin_capture.jl --train "Nome Pessoa" 15

# Com menos fotos (mínimo 10)
julia cnncheckin_capture.jl --train "Outro Nome" 10
```

#### Captura Incremental
```bash
# Adicionar nova pessoa
julia cnncheckin_capture.jl --incremental "Nova Pessoa" 10
```

#### Dicas de Captura

✅ **Faça**:
- Use boa iluminação (frontal ou lateral suave)
- Mantenha fundo neutro se possível
- Varie expressão facial entre capturas
- Varie levemente o ângulo da cabeça
- Mantenha distância de 50-100cm da câmera

❌ **Evite**:
- Óculos escuros
- Chapéus que cubram o rosto
- Iluminação muito forte atrás
- Movimento durante a captura
- Sombras fortes no rosto
- Reflexos em óculos
- Fotos muito distantes ou muito próximas

### 2. Treinamento Inicial

#### Modo Interativo (Recomendado)
```bash
julia cnncheckin_pretrain_webcam.jl
```

Opções:
1. Capturar novas imagens via webcam
2. Usar imagens existentes
3. Adicionar mais imagens E treinar

#### Modo Rápido
```bash
# Capturar e treinar múltiplas pessoas
julia cnncheckin_pretrain_webcam.jl --quick "Pessoa1" "Pessoa2" "Pessoa3" --num 15

# Apenas treinar (sem captura)
julia cnncheckin_pretrain_webcam.jl --no-capture
```

#### Parâmetros de Treinamento

O sistema usa os seguintes hiperparâmetros:
- **Epochs**: 30 (com early stopping)
- **Batch Size**: 8
- **Learning Rate**: 0.0001
- **Dropout**: 0.1-0.4 (por camada)
- **Data Augmentation**: Automático

**Tempo de treinamento**:
- 2-3 pessoas: ~5 minutos
- 4-5 pessoas: ~10 minutos
- 6-10 pessoas: ~15-20 minutos

### 3. Aprendizado Incremental

#### Modo Interativo
```bash
julia cnncheckin_incremental_webcam.jl
```

#### Modo Rápido
```bash
# Adicionar 2 novas pessoas
julia cnncheckin_incremental_webcam.jl --quick "Nova1" "Nova2" --num 10

# Treinar sem captura
julia cnncheckin_incremental_webcam.jl --no-capture
```

#### Vantagens do Aprendizado Incremental

✅ **Mais rápido**: Não retreina todo o modelo
✅ **Preserva conhecimento**: Mantém acurácia das pessoas antigas
✅ **Flexível**: Adicione pessoas quando necessário
✅ **Eficiente**: Usa Knowledge Distillation

**Tempo**: ~5 minutos por pessoa nova

### 4. Identificação

#### A. Identificação Única

```bash
# Modo interativo
julia cnncheckin_identify_webcam.jl

# Ou direto
julia cnncheckin_identify_webcam.jl --identify
```

Captura uma foto e identifica quem é a pessoa.

**Saída exemplo**:
```
✅ Pessoa identificada: João Silva
📊 Confiança: 94.2%
🔒 Nível de confiança: MUITO ALTA
```

#### B. Autenticação

```bash
# Verificar se é pessoa específica
julia cnncheckin_identify_webcam.jl --auth "João Silva" 0.75
```

Verifica se a pessoa é quem diz ser (útil para controle de acesso).

**Saída exemplo**:
```
✅ AUTENTICAÇÃO BEM-SUCEDIDA!
   Pessoa: João Silva
   Confiança: 87.3%
```

#### C. Modo Contínuo

```bash
# Identificar a cada 5 segundos
julia cnncheckin_identify_webcam.jl --continuous 5

# Com limite de tentativas
julia cnncheckin_identify_webcam.jl --continuous 10 20  # 10 seg, max 20 tentativas
```

Monitora continuamente e identifica pessoas.

**Útil para**:
- Monitoramento de sala
- Segurança
- Estatísticas de presença

#### D. Sistema Check-in/Check-out

```bash
# Sistema de registro de presença
julia cnncheckin_identify_webcam.jl --checkin presenca.csv
```

**Funcionalidades**:
- Detecta entrada (check-in) e saída (check-out)
- Mantém lista de pessoas presentes
- Gera log em CSV
- Timestamp de cada evento

**Arquivo de log** (presenca.csv):
```csv
2024-10-08 09:15:23,CHECK-IN,João Silva,0.9234
2024-10-08 09:47:12,CHECK-IN,Maria Santos,0.8876
2024-10-08 12:03:45,CHECK-OUT,João Silva,0.9156
2024-10-08 17:32:11,CHECK-OUT,Maria Santos,0.9023
```

## 🎯 Casos de Uso

### 1. Empresa - Controle de Acesso

```bash
# Setup inicial (primeiros funcionários)
julia cnncheckin_pretrain_webcam.jl --quick "João Silva" "Maria Santos" "Pedro Costa"

# Adicionar novos funcionários
julia cnncheckin_incremental_webcam.jl --quick "Ana Paula"

# Sistema na entrada
julia cnncheckin_identify_webcam.jl --checkin entrada_escritorio.csv
```

### 2. Escola - Registro de Presença

```bash
# Cadastrar alunos
julia cnncheckin_pretrain_webcam.jl --quick "Aluno1" "Aluno2" "Aluno3" --num 12

# Registrar presença em aula
julia cnncheckin_identify_webcam.jl --checkin aula_matematica.csv
```

### 3. Evento - Controle de Participantes

```bash
# Cadastrar participantes pré-inscritos
julia cnncheckin_pretrain_webcam.jl

# Durante o evento (identificação rápida)
julia cnncheckin_identify_webcam.jl --continuous 3
```

### 4. Residencial - Segurança

```bash
# Cadastrar moradores
julia cnncheckin_pretrain_webcam.jl --quick "Morador1" "Morador2"

# Autenticação na entrada
julia cnncheckin_identify_webcam.jl --auth "Morador1" 0.8
```

### 5. Academia - Check-in Automático

```bash
# Sistema de entrada
julia cnncheckin_identify_webcam.jl --checkin academia_checkin.csv

# Gera relatório automático de frequência
```

## 🔧 Configurações Avançadas

### Ajustar Parâmetros

Edite `cnncheckin_core.jl`:

```julia
# Tamanho das imagens (menor = mais rápido, maior = mais preciso)
const IMG_SIZE = (128, 128)  # Padrão: 128x128

# Batch size (maior = mais rápido em GPU, mais RAM)
const BATCH_SIZE = 8  # Padrão: 8

# Epochs de treinamento
const PRETRAIN_EPOCHS = 30  # Inicial
const INCREMENTAL_EPOCHS = 15  # Incremental

# Learning rates
const LEARNING_RATE = 0.0001
const INCREMENTAL_LR = 0.00005
```

### Múltiplas Câmeras

```bash
# Listar câmeras
julia cnncheckin_capture.jl --cameras

# Usar câmera específica (índice 1)
julia cnncheckin_identify_webcam.jl --identify 1
```

### Melhorar Acurácia

1. **Mais imagens por pessoa**:
   ```bash
   julia cnncheckin_capture.jl --train "Pessoa" 20  # 20 ao invés de 15
   ```

2. **Variar condições de captura**:
   - Diferentes iluminações
   - Diferentes expressões
   - Com/sem óculos (se usar)
   - Diferentes ângulos

3. **Retreinar periodicamente**:
   ```bash
   # A cada 2-3 meses, retreine com todas as imagens
   julia cnncheckin_pretrain_webcam.jl --no-capture
   ```

## 🐛 Troubleshooting

### Problema: Câmera não detectada

```bash
# Verificar câmeras
julia cnncheckin_capture.jl --cameras

# Testar câmera específica
julia cnncheckin_capture.jl --preview 0 5
```

**Soluções**:
- Verificar se webcam está conectada
- Fechar outros programas usando a câmera
- Verificar permissões (Linux/macOS)
- Reinstalar drivers (Windows)

### Problema: Erro ao instalar VideoIO

```bash
# Linux: instalar dependências
sudo apt-get install ffmpeg libavcodec-dev libavformat-dev libswscale-dev

# macOS
brew install ffmpeg

# Reinstalar pacote
julia -e 'using Pkg; Pkg.rm("VideoIO"); Pkg.add("VideoIO"); Pkg.build("VideoIO")'
```

### Problema: Baixa acurácia

**Causas comuns**:
1. Poucas imagens por pessoa (mínimo: 10, recomendado: 15+)
2. Imagens de baixa qualidade
3. Iluminação inconsistente
4. Variação excessiva (óculos, barba, etc.)

**Soluções**:
```bash
# Recapturar com mais fotos
julia cnncheckin_capture.jl --train "Pessoa" 20

# Retreinar
julia cnncheckin_pretrain_webcam.jl --no-capture
```

### Problema: Modelo muito lento

**GPU**: Se tiver GPU NVIDIA:
```julia
# Instalar CUDA
using Pkg
Pkg.add("CUDA")

# O sistema usará GPU automaticamente
```

**CPU**: Otimizações:
- Reduzir IMG_SIZE para (96, 96)
- Reduzir BATCH_SIZE para 4
- Usar menos imagens de treino

### Problema: Erro "out of memory"

```julia
# Reduzir batch size em cnncheckin_core.jl
const BATCH_SIZE = 4  # ao invés de 8

# Ou reduzir tamanho da imagem
const IMG_SIZE = (96, 96)  # ao invés de (128, 128)
```

### Problema: Pessoa não reconhecida

**Verificar**:
1. Pessoa está no modelo?
   ```bash
   # Ver pessoas cadastradas
   julia -e 'using TOML; config = TOML.parsefile("face_recognition_config.toml"); println(config["data"]["person_names"])'
   ```

2. Iluminação similar ao treinamento?
3. Distância similar?

**Solução**: Adicionar mais fotos em condições variadas

## 📊 Estatísticas e Monitoramento

### Ver Informações do Modelo

```bash
# Abrir Julia REPL
julia

# Carregar configuração
using TOML
config = TOML.parsefile("face_recognition_config.toml")

# Ver pessoas
println("Pessoas: ", config["data"]["person_names"])

# Ver acurácia
println("Acurácia: ", config["training"]["final_accuracy"])

# Ver histórico
model_data = TOML.parsefile("face_recognition_model_data.toml")
println("Últimas predições: ", model_data["prediction_examples"])
```

### Analisar Log de Check-in

```julia
using CSV, DataFrames

# Ler log
df = CSV.read("presenca.csv", DataFrame, 
              header=["timestamp", "acao", "pessoa", "confianca"])

# Estatísticas
println("Total de registros: ", nrow(df))
println("Pessoas únicas: ", length(unique(df.pessoa)))

# Agrupar por pessoa
using Statistics
by_person = groupby(df, :pessoa)
combine(by_person, nrow => :total)
```

## 🔐 Segurança e Privacidade

### Recomendações

1. **Consentimento**: Obtenha autorização antes de cadastrar pessoas
2. **Armazenamento**: Proteja o arquivo do modelo
3. **Logs**: Defina política de retenção
4. **Acesso**: Restrinja quem pode treinar/identificar
5. **LGPD/GDPR**: Siga regulamentações locais

### Proteger Modelo

```bash
# Linux/macOS: restringir permissões
chmod 600 face_recognition_model.jld2
chmod 600 face_recognition_config.toml

# Backup seguro
tar -czf backup_modelo.tar.gz face_recognition_*.jld2 face_recognition_*.toml
gpg -c backup_modelo.tar.gz  # Criptografar
```

## 🚀 Performance

### Benchmarks (CPU i5, 16GB RAM)

| Operação | Tempo |
|----------|-------|
| Captura de imagem | ~1s |
| Pré-processamento | ~0.1s |
| Identificação (CPU) | ~0.3s |
| Identificação (GPU) | ~0.05s |
| Treinamento inicial (3 pessoas) | ~5min |
| Treinamento incremental (1 pessoa) | ~3min |

### Otimizações

**Para GPU NVIDIA**:
```bash
# Habilitar CUDA
export CUDA_VISIBLE_DEVICES=0
julia --project
```

**Para múltiplos cores**:
```julia
# Adicionar threads
julia -t 4  # 4 threads

# O Flux usará automaticamente
```

## 📚 Recursos Adicionais

### Documentação
- [Flux.jl Docs](https://fluxml.ai/Flux.jl/stable/)
- [Images.jl Docs](https://juliaimages.org/stable/)
- [VideoIO.jl Docs](https://juliaio.github.io/VideoIO.jl/stable/)

### Comunidade
- [Julia Discourse](https://discourse.julialang.org/)
- [Flux Slack](https://julialang.org/slack/)

### Tutoriais
- [Deep Learning com Julia](https://fluxml.ai/tutorials/)
- [Processamento de Imagens](https://juliaimages.org/stable/tutorials/)

## 🤝 Contribuindo

Melhorias são bem-vindas! Áreas de interesse:
- Suporte a reconhecimento de múltiplas faces
- Interface gráfica (GTK/Qt)
- API REST
- Detecção de liveness (anti-spoofing)
- Reconhecimento por máscara

## 📄 Licença

Este projeto é fornecido "como está" para fins educacionais e de pesquisa.

## ⚠️ Avisos Legais

- **Uso Responsável**: Este sistema deve ser usado de forma ética
- **Precisão**: Não é 100% preciso, não use para decisões críticas
- **Privacidade**: Respeite leis de proteção de dados
- **Consentimento**: Obtenha permissão antes de cadastrar pessoas
- **Bias**: Modelos podem ter viés, teste com diversidade

## 📞 Suporte

Para problemas ou dúvidas:
1. Verifique este README
2. Consulte o Troubleshooting
3. Procure em Julia Discourse
4. Abra uma issue no repositório

---

**Versão**: 2.0 com Webcam  
**Última atualização**: Outubro 2024  
**Compatibilidade**: Julia 1.9+

🎉 **Bom uso do CNNCheckin!** 🎉