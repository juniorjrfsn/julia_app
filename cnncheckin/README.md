# 🎥 CNNCheckin - Sistema de Reconhecimento Facial com Webcam

Sistema completo de reconhecimento facial usando Deep Learning (CNN) com Julia, incluindo captura via webcam integrada (compatível com Iriun Webcam).

## ⚡ Início Rápido (3 Comandos)

```bash
# 1. Instalar dependências
cd ~/Documentos/projetos/julia_app/cnncheckin/src
julia --project -e 'using Pkg; Pkg.add(["Flux", "Images", "VideoIO", "ImageView", "JLD2", "TOML", "ImageTransformations"])'

# 2. Treinar modelo com 3 pessoas (15 fotos cada)
julia cnncheckin_pretrain_webcam.jl --quick "João" "Maria" "Pedro" --num 15

# 3. Identificar pessoas
julia cnncheckin_identify_webcam.jl --identify
```

---

## 📋 Índice

- [Características](#-características)
- [Requisitos](#-requisitos)
- [Instalação](#-instalação)
- [Uso Básico](#-uso-básico)
- [Comandos Disponíveis](#-comandos-disponíveis)
- [Casos de Uso](#-casos-de-uso)
- [Troubleshooting](#-troubleshooting)
- [Referência Rápida](#-referência-rápida)

---

## ✨ Características

### 🎯 Funcionalidades Principais

- **✅ Captura via Webcam** - Captura imagens diretamente da câmera (incluindo Iriun Webcam)
- **✅ Treinamento Inicial** - Treina modelo CNN com múltiplas pessoas
- **✅ Aprendizado Incremental** - Adiciona novas pessoas sem retreinar tudo
- **✅ Identificação em Tempo Real** - Identifica pessoas instantaneamente
- **✅ Autenticação** - Verifica identidade de pessoa específica
- **✅ Modo Contínuo** - Monitoramento constante
- **✅ Check-in/Check-out** - Sistema de presença automático

### 🔧 Tecnologias

- **Julia 1.9+** - Linguagem de alta performance
- **Flux.jl** - Deep Learning framework
- **VideoIO.jl** - Captura de vídeo e webcam
- **Images.jl** - Processamento de imagens
- **CNN Personalizada** - Rede neural convolucional otimizada

---

## 📦 Requisitos

### Sistema Operacional

- ✅ Linux
- ✅ macOS
- ✅ Windows (com WSL recomendado)
- 📷 Webcam conectada (ou Iriun Webcam no celular)

### Software

```julia
# Julia 1.9 ou superior
julia --version

# Pacotes necessários
Flux, Images, FileIO, VideoIO, ImageView
Statistics, Random, JLD2, TOML
ImageTransformations, LinearAlgebra, Dates
```

### Hardware

- **Mínimo**: CPU 2+ cores, 4GB RAM, Webcam
- **Recomendado**: CPU 4+ cores, 8GB RAM, GPU NVIDIA (opcional)

---

## 🚀 Instalação

### 1. Instalar Julia

```bash
# Linux/macOS - Julia 1.10.0
wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.0-linux-x86_64.tar.gz
tar -xvzf julia-1.10.0-linux-x86_64.tar.gz
sudo mv julia-1.10.0 /opt/
sudo ln -s /opt/julia-1.10.0/bin/julia /usr/local/bin/julia

# Verificar instalação
julia --version
```

### 2. Criar Estrutura do Projeto

```bash
cd ~/Documentos/projetos/julia_app
mkdir -p cnncheckin/{src,dados/{fotos_train,fotos_new,fotos_auth}}
cd cnncheckin/src
```

### 3. Instalar Dependências Julia

```bash
# Opção 1: Instalação rápida
julia --project -e 'using Pkg; Pkg.add(["Flux", "Images", "FileIO", "VideoIO", "ImageView", "JLD2", "TOML", "ImageTransformations", "Statistics", "Random", "LinearAlgebra", "Dates"])'

# Opção 2: Instalação interativa
julia
```

```julia
# Dentro do Julia REPL
using Pkg
Pkg.activate(".")

# Instalar pacotes
Pkg.add([
    "Flux",
    "Images", 
    "FileIO",
    "VideoIO",
    "ImageView",
    "JLD2",
    "TOML",
    "ImageTransformations"
])

# Verificar instalação
using Flux, Images, VideoIO
println("✅ Pacotes instalados com sucesso!")
exit()
```

### 4. Configurar Iriun Webcam (se usar celular)

```bash
# Linux
# 1. Baixar Iriun Webcam no celular (Android/iOS)
# 2. Instalar driver no computador:
wget http://iriun.com/downloads/iriun-webcam-linux-2.8.2.deb
sudo dpkg -i iriun-webcam-linux-2.8.2.deb

# 3. Iniciar Iriun no celular e conectar (USB ou WiFi)
# 4. Verificar dispositivo:
ls -l /dev/video*

# Windows/macOS
# Baixar instalador em: http://iriun.com
```

### 5. Testar Câmera

```bash
julia cnncheckin_capture.jl --cameras
```

**Saída esperada:**

```
🎥 Detectando câmeras disponíveis...
  ✔ Câmera 0 detectada
  ✔ Câmera 2 detectada (Iriun Webcam)
✅ Total de câmeras encontradas: 2
```

---

## 🎯 Uso Básico

### Fluxo Completo em 3 Passos

#### 1️⃣ **Treinar Modelo Inicial**

```bash
# Modo rápido: capturar 3 pessoas com 15 fotos cada
julia cnncheckin_pretrain_webcam.jl --quick "João Silva" "Maria Santos" "Pedro Costa" --num 15
```

**O que acontece:**

- ✅ Captura 15 fotos de cada pessoa
- ✅ Treina modelo CNN automaticamente  
- ✅ Salva modelo e configurações
- ⏱️ **Tempo:** 10-15 minutos

#### 2️⃣ **Adicionar Novas Pessoas**

```bash
# Adicionar 2 novas pessoas com 10 fotos cada
julia cnncheckin_incremental_webcam.jl --quick "Carlos Alberto" "Ana Paula" --num 10
```

**O que acontece:**

- ✅ Captura fotos das novas pessoas
- ✅ Treina incrementalmente (rápido!)
- ✅ Mantém pessoas anteriores
- ⏱️ **Tempo:** 5-8 minutos

#### 3️⃣ **Identificar Pessoas**

```bash
# Identificação única
julia cnncheckin_identify_webcam.jl --identify
```

**Resultado:**

```
✅ Pessoa identificada: João Silva
📊 Confiança: 94.2%
🔒 Nível: MUITO ALTA
```

---

## 📖 Comandos Disponíveis

### 🎓 Treinamento Inicial

```bash
# Menu interativo (recomendado para iniciantes)
julia cnncheckin_pretrain_webcam.jl

# Modo rápido - 3 pessoas, 15 fotos cada
julia cnncheckin_pretrain_webcam.jl --quick "Pessoa1" "Pessoa2" "Pessoa3" --num 15

# Treinar sem capturar (usar imagens existentes)
julia cnncheckin_pretrain_webcam.jl --no-capture

# Ver ajuda
julia cnncheckin_pretrain_webcam.jl --help
```

### 📚 Aprendizado Incremental

```bash
# Menu interativo
julia cnncheckin_incremental_webcam.jl

# Modo rápido - adicionar 2 pessoas, 10 fotos cada
julia cnncheckin_incremental_webcam.jl --quick "Nova1" "Nova2" --num 10

# Treinar sem captura
julia cnncheckin_incremental_webcam.jl --no-capture

# Ver ajuda
julia cnncheckin_incremental_webcam.jl --help
```

### 🎯 Identificação

```bash
# Menu interativo
julia cnncheckin_identify_webcam.jl

# Identificação única
julia cnncheckin_identify_webcam.jl --identify

# Autenticação (verificar pessoa específica)
julia cnncheckin_identify_webcam.jl --auth "João Silva" 0.75

# Modo contínuo (identificar a cada 5 segundos)
julia cnncheckin_identify_webcam.jl --continuous 5

# Sistema de check-in/check-out
julia cnncheckin_identify_webcam.jl --checkin presenca.csv

# Ver ajuda
julia cnncheckin_identify_webcam.jl --help
```

### 📸 Captura Manual

```bash
# Menu interativo
julia cnncheckin_capture.jl

# Capturar para treinamento inicial
julia cnncheckin_capture.jl --train "Nome Pessoa" 15

# Capturar para aprendizado incremental
julia cnncheckin_capture.jl --incremental "Nova Pessoa" 10

# Listar câmeras disponíveis
julia cnncheckin_capture.jl --cameras

# Preview da câmera por 5 segundos
julia cnncheckin_capture.jl --preview 0 5

# Usar câmera específica (Iriun geralmente é índice 2)
julia cnncheckin_capture.jl --train "Nome" 15 --camera 2

# Ver ajuda
julia cnncheckin_capture.jl --help
```

---

## 💡 Casos de Uso

### 1. 🏢 Empresa - Controle de Acesso

```bash
# Setup inicial (primeiros funcionários)
julia cnncheckin_pretrain_webcam.jl --quick "João Silva" "Maria Santos" "Pedro Costa"

# Adicionar novos funcionários
julia cnncheckin_incremental_webcam.jl --quick "Ana Paula"

# Sistema na entrada (portaria)
julia cnncheckin_identify_webcam.jl --checkin entrada_escritorio.csv
```

**Resultado:** Sistema automático de registro de entrada/saída

### 2. 🎓 Escola - Registro de Presença

```bash
# Cadastrar alunos (uma vez)
julia cnncheckin_pretrain_webcam.jl --quick "Aluno1" "Aluno2" "Aluno3" --num 12

# Registrar presença na aula
julia cnncheckin_identify_webcam.jl --checkin aula_matematica.csv
```

**Resultado:** Chamada automática instantânea

### 3. 🎪 Evento - Controle de Participantes

```bash
# Cadastrar participantes pré-inscritos
julia cnncheckin_pretrain_webcam.jl

# Durante o evento (identificação contínua)
julia cnncheckin_identify_webcam.jl --continuous 3
```

**Resultado:** Monitoramento em tempo real

### 4. 🏠 Residencial - Segurança

```bash
# Cadastrar moradores
julia cnncheckin_pretrain_webcam.jl --quick "Morador1" "Morador2"

# Autenticação na entrada
julia cnncheckin_identify_webcam.jl --auth "Morador1" 0.8
```

**Resultado:** Controle de acesso seguro

### 5. 💪 Academia - Check-in Automático

```bash
# Sistema de entrada
julia cnncheckin_identify_webcam.jl --checkin academia_checkin.csv
```

**Resultado:** Relatório automático de frequência

---

## 🛠 Troubleshooting

### ❌ Problema: Erro de carregamento CUDA/GPU

**Erro:**

```
Error during loading of extension FluxCUDAExt
ConcurrencyViolationError: deadlock detected
```

**Solução:**

```bash
# Desabilitar CUDA temporariamente
export JULIA_CUDA_USE_BINARYBUILDER=false

# Ou remover CUDA (se não tiver GPU NVIDIA)
julia -e 'using Pkg; Pkg.rm("CUDA")'
```

### ❌ Problema: Câmera não detectada (Iriun Webcam)

**Verificar:**

```bash
# Listar câmeras
julia cnncheckin_capture.jl --cameras

# Verificar dispositivos (Linux)
ls -l /dev/video*
v4l2-ctl --list-devices

# Testar câmera específica
julia cnncheckin_capture.jl --preview 2 5  # Tente índices 0-10
```

**Soluções:**

```bash
# Linux: adicionar usuário ao grupo video
sudo usermod -a -G video $USER
# Fazer logout/login

# Verificar se Iriun está rodando
ps aux | grep iriun
sudo systemctl status iriunwebcam

# Reiniciar serviço Iriun
sudo systemctl restart iriunwebcam

# Verificar permissões
sudo chmod 666 /dev/video*

# Fechar outros programas (Zoom, Skype, etc.)
```

**Solução alternativa com Python (mais confiável):**

```bash
# Instalar OpenCV para Python
pip3 install opencv-python

# Usar script Python auxiliar
python3 capture_opencv.py --test
python3 capture_opencv.py --single foto.jpg --camera 2
```

### ❌ Problema: Erro ao instalar VideoIO

**Solução Linux:**

```bash
# Instalar dependências
sudo apt-get update
sudo apt-get install ffmpeg libavcodec-dev libavformat-dev libswscale-dev

# Reinstalar pacote
julia -e 'using Pkg; Pkg.rm("VideoIO"); Pkg.add("VideoIO"); Pkg.build("VideoIO")'
```

**Solução macOS:**

```bash
brew install ffmpeg
julia -e 'using Pkg; Pkg.build("VideoIO")'
```

### ❌ Problema: Modelo não carrega

**Verificar:**

```bash
# Verificar existência
ls -lh face_recognition_model.jld2

# Verificar configuração
cat face_recognition_config.toml
```

**Solução:**

```bash
# Retreinar se corrompido
julia cnncheckin_pretrain_webcam.jl --no-capture
```

### ❌ Problema: Baixa acurácia

**Causas comuns:**

- ❌ Poucas imagens por pessoa (mínimo: 10, recomendado: 15+)
- ❌ Imagens de baixa qualidade
- ❌ Iluminação inconsistente
- ❌ Variação excessiva (óculos, barba, etc.)

**Soluções:**

```bash
# 1. Recapturar com mais fotos
julia cnncheckin_capture.jl --train "Pessoa" 20

# 2. Retreinar
julia cnncheckin_pretrain_webcam.jl --no-capture

# 3. Melhorar iluminação durante captura
# 4. Capturar em condições similares à identificação
```

### ❌ Problema: Modelo muito lento

**Para GPU (se tiver NVIDIA):**

```julia
using Pkg
Pkg.add("CUDA")
# O sistema usará GPU automaticamente
```

**Para CPU - Otimizações:**

Edite `cnncheckin_core.jl`:

```julia
# Reduzir tamanho da imagem
const IMG_SIZE = (96, 96)  # ao invés de (128, 128)

# Reduzir batch size
const BATCH_SIZE = 4  # ao invés de 8
```

---

## 🚀 Referência Rápida de Comandos

### 📸 Captura de Imagens

| Comando | Descrição |
|---------|-----------|
| `julia cnncheckin_capture.jl` | Menu interativo |
| `julia cnncheckin_capture.jl --train "Nome" 15` | Capturar para treinamento inicial (15 fotos) |
| `julia cnncheckin_capture.jl --incremental "Nome" 10` | Capturar para adicionar pessoa (10 fotos) |
| `julia cnncheckin_capture.jl --identify` | Capturar para identificação |
| `julia cnncheckin_capture.jl --cameras` | Listar câmeras disponíveis |
| `julia cnncheckin_capture.jl --preview 0 5` | Preview da câmera 0 por 5 segundos |
| `julia cnncheckin_capture.jl --train "Nome" 15 --camera 2` | Usar câmera específica (Iriun) |

### 🎓 Treinamento Inicial

| Comando | Descrição |
|---------|-----------|
| `julia cnncheckin_pretrain_webcam.jl` | Menu interativo |
| `julia cnncheckin_pretrain_webcam.jl --quick "P1" "P2" --num 15` | Modo rápido |
| `julia cnncheckin_pretrain_webcam.jl --no-capture` | Treinar sem captura |

**Parâmetros Padrão:**

- Imagens por pessoa: 15
- Epochs: 30 (com early stopping)
- Batch size: 8
- Learning rate: 0.0001

### 📚 Aprendizado Incremental

| Comando | Descrição |
|---------|-----------|
| `julia cnncheckin_incremental_webcam.jl` | Menu interativo |
| `julia cnncheckin_incremental_webcam.jl --quick "Nova" --num 10` | Modo rápido |
| `julia cnncheckin_incremental_webcam.jl --no-capture` | Treinar sem captura |

**Parâmetros Padrão:**

- Imagens por pessoa: 10
- Epochs: 15
- Learning rate: 0.00005

### 🎯 Identificação

| Modo | Comando |
|------|---------|
| **Identificação única** | `julia cnncheckin_identify_webcam.jl --identify` |
| **Autenticação** | `julia cnncheckin_identify_webcam.jl --auth "Nome" 0.7` |
| **Contínuo** | `julia cnncheckin_identify_webcam.jl --continuous 5` |
| **Check-in** | `julia cnncheckin_identify_webcam.jl --checkin presenca.csv` |

### 🎨 Personalização

```julia
# Edite cnncheckin_core.jl:

# Tamanho da imagem
const IMG_SIZE = (128, 128)  # Padrão: 128x128

# Batch size
const BATCH_SIZE = 8  # Padrão: 8

# Epochs
const PRETRAIN_EPOCHS = 30
const INCREMENTAL_EPOCHS = 15

# Learning rates
const LEARNING_RATE = 0.0001
const INCREMENTAL_LR = 0.00005
```

### 🔧 Aliases Úteis

Adicione ao `~/.bashrc` ou `~/.zshrc`:

```bash
# Aliases CNNCheckin
alias cnn-capture='cd ~/Documentos/projetos/julia_app/cnncheckin/src && julia cnncheckin_capture.jl'
alias cnn-train='cd ~/Documentos/projetos/julia_app/cnncheckin/src && julia cnncheckin_pretrain_webcam.jl'
alias cnn-add='cd ~/Documentos/projetos/julia_app/cnncheckin/src && julia cnncheckin_incremental_webcam.jl'
alias cnn-identify='cd ~/Documentos/projetos/julia_app/cnncheckin/src && julia cnncheckin_identify_webcam.jl'
alias cnn-checkin='cd ~/Documentos/projetos/julia_app/cnncheckin/src && julia cnncheckin_identify_webcam.jl --checkin'
alias cnn-backup='cd ~/Documentos/projetos/julia_app/cnncheckin/src && tar -czf ../backup_$(date +%Y%m%d).tar.gz face_recognition_*'
```

---

## 📊 Estatísticas e Monitoramento

### Ver Configuração do Modelo

```bash
julia -e 'using TOML; config = TOML.parsefile("face_recognition_config.toml"); 
          println("Pessoas: ", config["data"]["person_names"]); 
          println("Acurácia: ", config["training"]["final_accuracy"])'
```

### Análise de Log CSV

```bash
# Ver últimas entradas
tail -20 presenca.csv

# Contar registros
wc -l presenca.csv

# Filtrar por pessoa
grep "João Silva" presenca.csv
```

### Backup do Modelo

```bash
# Backup completo
tar -czf backup_modelo_$(date +%Y%m%d).tar.gz face_recognition_*.jld2 face_recognition_*.toml

# Restaurar backup
tar -xzf backup_modelo_20241008.tar.gz
```

---

## 🎯 Casos de Uso por Comando

| Cenário | Comando |
|---------|---------|
| Primeira instalação | `julia cnncheckin_pretrain_webcam.jl` |
| Novo funcionário | `julia cnncheckin_incremental_webcam.jl --quick "Nome"` |
| Controle de acesso | `julia cnncheckin_identify_webcam.jl --auth "Nome" 0.75` |
| Monitoramento | `julia cnncheckin_identify_webcam.jl --continuous 10` |
| Registro de presença | `julia cnncheckin_identify_webcam.jl --checkin presenca.csv` |
| Teste rápido | `julia cnncheckin_identify_webcam.jl --identify` |
| Usar Iriun Webcam | Adicione `--camera 2` aos comandos |

---

## 📁 Estrutura do Projeto

```
cnncheckin/
├── src/
│   ├── cnncheckin_core.jl                    # Módulo central
│   ├── cnncheckin_webcam.jl                  # Módulo webcam
│   ├── cnncheckin_capture.jl                 # Interface de captura
│   ├── capture_iriun.jl                      # Captura Iriun específica
│   ├── cnncheckin_pretrain_webcam.jl         # Treinamento + webcam
│   ├── cnncheckin_incremental_webcam.jl      # Incremental + webcam
│   ├── cnncheckin_identify_webcam.jl         # Identificação + webcam
│   ├── face_recognition_model.jld2           # Modelo treinado (gerado)
│   ├── face_recognition_config.toml          # Configuração (gerado)
│   └── face_recognition_model_data.toml      # Metadados (gerado)
├── dados/
│   ├── fotos_train/                          # Imagens treinamento
│   ├── fotos_new/                            # Imagens incrementais
│   └── fotos_auth/                           # Imagens identificação
├── README.md                                  # Este arquivo
└── Project.toml                               # Configuração Julia
```

---

## 📝 Dicas de Uso Diário

### ✅ Boas Práticas

**Faça:**

- ✅ Backup semanal do modelo
- ✅ Mantenha boa iluminação
- ✅ Capture mínimo 10 fotos/pessoa
- ✅ Varie expressões e ângulos
- ✅ Retreine a cada 2-3 meses

**Evite:**

- ❌ Capturar com pouca luz
- ❌ Usar óculos escuros
- ❌ Movimentar durante captura
- ❌ Adicionar muitas pessoas de uma vez
- ❌ Ignorar avisos de confiança baixa

### Workflow Recomendado

**Setup Inicial (uma vez):**

```bash
1. julia cnncheckin_capture.jl --train "Pessoa1" 15
2. julia cnncheckin_capture.jl --train "Pessoa2" 15
3. julia cnncheckin_pretrain_webcam.jl --no-capture
```

**Adicionar Pessoa (quando necessário):**

```bash
1. julia cnncheckin_capture.jl --incremental "NovaPessoa" 10
2. julia cnncheckin_incremental_webcam.jl --no-capture
```

**Uso Diário:**

```bash
# Sistema de entrada/saída
julia cnncheckin_identify_webcam.jl --checkin presenca_diaria.csv
```

---

## 🔒 Segurança e Privacidade

### Recomendações

1. **✅ Consentimento** - Obtenha autorização antes de cadastrar pessoas
2. **✅ Armazenamento** - Proteja o arquivo do modelo
3. **✅ Logs** - Defina política de retenção
4. **✅ Acesso** - Restrinja quem pode treinar/identificar
5. **✅ LGPD/GDPR** - Siga regulamentações locais

### Proteger Modelo

```bash
# Linux/macOS: restringir permissões
chmod 600 face_recognition_model.jld2
chmod 600 face_recognition_config.toml

# Backup seguro com criptografia
tar -czf backup_modelo.tar.gz face_recognition_*.jld2 face_recognition_*.toml
gpg -c backup_modelo.tar.gz  # Criptografar (pedirá senha)

# Descriptografar
gpg backup_modelo.tar.gz.gpg
```

---

## ⚠️ Avisos Legais

- **Uso Responsável**: Este sistema deve ser usado de forma ética
- **Precisão**: Não é 100% preciso, não use para decisões críticas
- **Privacidade**: Respeite leis de proteção de dados
- **Consentimento**: Obtenha permissão antes de cadastrar pessoas
- **Bias**: Modelos podem ter viés, teste com diversidade

---

## 📞 Suporte

Para problemas ou dúvidas:

1. ✅ Verifique este README
2. ✅ Consulte o [Troubleshooting](#-troubleshooting)
3. ✅ Procure em [Julia Discourse](https://discourse.julialang.org/)

---

**Versão**: 2.0 com Webcam (compatível Iriun)  
**Última atualização**: Outubro 2024  
**Compatibilidade**: Julia 1.9+

🎉 **Bom uso do CNNCheckin!** 🎉
