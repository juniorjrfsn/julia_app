# 🚀 Guia de Instalação Rápida - CNNCheckin

Instale e configure o sistema em menos de 15 minutos!

## 📋 Pré-requisitos

- Sistema operacional: Linux, macOS ou Windows
- Webcam funcional
- Conexão com internet
- ~2GB de espaço em disco

## ⚡ Instalação Rápida

### 1. Instalar Julia (5 minutos)

#### Linux
```bash
# Baixar Julia 1.9.4
wget https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.4-linux-x86_64.tar.gz

# Extrair
tar -xvzf julia-1.9.4-linux-x86_64.tar.gz

# Mover para /opt
sudo mv julia-1.9.4 /opt/

# Criar link simbólico
sudo ln -s /opt/julia-1.9.4/bin/julia /usr/local/bin/julia

# Verificar instalação
julia --version
```

#### macOS
```bash
# Com Homebrew
brew install julia

# Ou baixar do site oficial
# https://julialang.org/downloads/
```

#### Windows
```powershell
# Baixar instalador de:
# https://julialang.org/downloads/

# Executar o instalador
# Adicionar ao PATH quando solicitado
```

### 2. Instalar Dependências do Sistema (2 minutos)

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libavcodec-dev libavformat-dev libswscale-dev \
                        libgtk-3-dev libcairo2-dev libpango1.0-dev
```

#### macOS
```bash
brew install ffmpeg gtk+3
```

#### Windows
```powershell
# FFmpeg será instalado automaticamente pelo Julia
# Ou baixe de: https://ffmpeg.org/download.html
```

### 3. Criar Estrutura do Projeto (1 minuto)

```bash
# Criar diretórios
mkdir -p cnncheckin/src
mkdir -p cnncheckin/dados/{fotos_train,fotos_new,fotos_auth}

# Navegar para o projeto
cd cnncheckin
```

### 4. Instalar Pacotes Julia (5 minutos)

```bash
cd src

# Criar e ativar ambiente
julia --project -e 'using Pkg; Pkg.activate(".")'

# Instalar pacotes
julia --project << 'EOF'
using Pkg

# Pacotes principais
pacotes = [
    "Flux",
    "Images",
    "FileIO",
    "CUDA",
    "Statistics",
    "Random",
    "JLD2",
    "TOML",
    "ImageTransformations",
    "LinearAlgebra",
    "Dates",
    "ImageView",
    "VideoIO"
]

println("📦 Instalando pacotes...")
for pkg in pacotes
    try
        println("   Instalando $pkg...")
        Pkg.add(pkg)
    catch e
        println("   ⚠️  Erro ao instalar $pkg: $e")
    end
end

println("\n🔨 Compilando pacotes...")
Pkg.build()

println("\n✅ Instalação concluída!")
EOF
```

### 5. Copiar Arquivos do Sistema

Copie os seguintes arquivos para `cnncheckin/src/`:

**Arquivos necessários:**
- `cnncheckin_core.jl`
- `cnncheckin_webcam.jl`
- `cnncheckin_capture.jl`
- `cnncheckin_pretrain.jl`
- `cnncheckin_pretrain_webcam.jl`
- `cnncheckin_incremental.jl`
- `cnncheckin_incremental_webcam.jl`
- `cnncheckin_identify.jl`
- `cnncheckin_identify_webcam.jl`

### 6. Testar Instalação (2 minutos)

```bash
cd src

# Testar Julia
julia --version

# Testar importação de pacotes
julia --project << 'EOF'
using Flux
using Images
using VideoIO
println("✅ Todos os pacotes carregados com sucesso!")
EOF

# Testar câmera
julia cnncheckin_capture.jl --cameras
```

## ✅ Verificação da Instalação

Execute este script de teste:

```bash
cd src
julia --project << 'EOF'
println("🧪 Testando instalação do CNNCheckin\n")

# Teste 1: Pacotes
println("📦 Teste 1: Verificando pacotes...")
try
    using Flux, Images, VideoIO, JLD2, TOML
    println("   ✅ Todos os pacotes OK")
catch e
    println("   ❌ Erro: $e")
    exit(1)
end

# Teste 2: Estrutura de diretórios
println("\n📁 Teste 2: Verificando diretórios...")
dirs = ["../dados/fotos_train", "../dados/fotos_new", "../dados/fotos_auth"]
for dir in dirs
    if isdir(dir)
        println("   ✅ $dir")
    else
        println("   ❌ $dir não encontrado")
    end
end

# Teste 3: Arquivos do sistema
println("\n📄 Teste 3: Verificando arquivos...")
files = [
    "cnncheckin_core.jl",
    "cnncheckin_webcam.jl",
    "cnncheckin_capture.jl"
]
for file in files
    if isfile(file)
        println("   ✅ $file")
    else
        println("   ⚠️  $file não encontrado")
    end
end

println("\n🎉 Instalação verificada com sucesso!")
println("\n📖 Próximos passos:")
println("   1. Testar câmera: julia cnncheckin_capture.jl --cameras")
println("   2. Ver tutorial: cat ../README_WEBCAM.md")
println("   3. Executar exemplo: bash exemplo_completo.sh")
EOF
```

## 🐛 Solução de Problemas Comuns

### Erro: "VideoIO not found"

**Linux:**
```bash
sudo apt-get install -y ffmpeg libavcodec-dev libavformat-dev libswscale-dev
julia --project -e 'using Pkg; Pkg.build("VideoIO")'
```

**macOS:**
```bash
brew install ffmpeg
julia --project -e 'using Pkg; Pkg.build("VideoIO")'
```

**Windows:**
- Baixe FFmpeg de https://ffmpeg.org/download.html
- Adicione ao PATH do sistema
- Reinicie Julia

### Erro: "Cannot open camera"

```bash
# Linux: adicionar usuário ao grupo video
sudo usermod -a -G video $USER
# Fazer logout e login novamente

# Verificar permissões
ls -l /dev/video*

# Testar com outros programas
cheese  # ou vlc, ou guvcview
```

### Erro: "Out of memory"

Edite `cnncheckin_core.jl`:
```julia
# Reduzir uso de memória
const IMG_SIZE = (96, 96)  # ao invés de (128, 128)
const BATCH_SIZE = 4       # ao invés de 8
```

### Erro: "Package precompilation failed"

```bash
# Limpar cache e reinstalar
julia --project -e 'using Pkg; Pkg.rm("Flux"); Pkg.gc(); Pkg.add("Flux")'

# Ou remover todo o ambiente
rm -rf ~/.julia/compiled
julia --project -e 'using Pkg; Pkg.build()'
```

## 🎯 Primeiro Uso

Após a instalação:

```bash
cd src

# 1. Testar câmera
julia cnncheckin_capture.jl --preview 0 5

# 2. Capturar primeira pessoa
julia cnncheckin_capture.jl --train "Seu Nome" 15

# 3. Treinar modelo
julia cnncheckin_pretrain_webcam.jl --no-capture

# 4. Testar identificação
julia cnncheckin_identify_webcam.jl --identify
```

## 📦 Instalação com Docker (Alternativa)

Se preferir usar Docker:

```dockerfile
# Dockerfile
FROM julia:1.9

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# Criar diretório de trabalho
WORKDIR /app

# Copiar arquivos
COPY src/ /app/src/
COPY dados/ /app/dados/

# Instalar pacotes Julia
WORKDIR /app/src
RUN julia --project -e 'using Pkg; \
    Pkg.add(["Flux", "Images", "VideoIO", "ImageView", "JLD2", "TOML"]); \
    Pkg.build()'

# Expor webcam (adicionar ao docker run: --device=/dev/video0)
CMD ["bash"]
```

```bash
# Construir imagem
docker build -t cnncheckin .

# Executar (Linux)
docker run -it --rm \
    --device=/dev/video0 \
    -v $(pwd)/dados:/app/dados \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    cnncheckin

# Dar acesso ao X11 (Linux)
xhost +local:docker
```

## ⚙️ Configuração Avançada

### Para GPU NVIDIA

```bash
# Instalar CUDA Toolkit
# https://developer.nvidia.com/cuda-downloads

# Verificar instalação
nvidia-smi

# Configurar Julia para usar GPU
julia --project -e 'using Pkg; Pkg.add("CUDA"); using CUDA; CUDA.functional()'

# O sistema usará GPU automaticamente
```

### Para Múltiplas Câmeras

```bash
# Listar câmeras
julia cnncheckin_capture.jl --cameras

# Usar câmera específica (índice 1)
export CAMERA_INDEX=1
# Ou passar como argumento aos scripts
```

### Otimização de Performance

Edite `~/.julia/config/startup.jl`:
```julia
# Usar múltiplos threads
ENV["JULIA_NUM_THREADS"] = "4"

# Pre-compilar pacotes comuns
using Flux, Images
```

## 📊 Verificar Instalação Completa

Checklist final:

- [ ] Julia 1.9+ instalado
- [ ] FFmpeg instalado
- [ ] Todos os pacotes Julia instalados
- [ ] Estrutura de diretórios criada
- [ ] Arquivos .jl copiados
- [ ] Câmera detectada e funcionando
- [ ] Teste de captura bem-sucedido

## 🎓 Recursos de Aprendizado

Após a instalação:

1. **Tutorial Básico**: `README_WEBCAM.md`
2. **Exemplo Completo**: `exemplo_completo.sh`
3. **Documentação Julia**: https://docs.julialang.org/
4. **Flux.jl Tutorial**: https://fluxml.ai/tutorials/

## 🆘 Suporte

Se encontrar problemas:

1. Verifique a seção Troubleshooting no README_WEBCAM.md
2. Consulte Julia Discourse: https://discourse.julialang.org/
3. Verifique logs de erro: `julia --project --trace-compile=stderr`
4. Teste componentes individuais

## 📞 Precisa de Ajuda?

```bash
# Verificar versões
julia --version
ffmpeg -version

# Gerar relatório de sistema
julia --project << 'EOF'
using InteractiveUtils
versioninfo()
EOF

# Testar pacotes individualmente
julia --project -e 'using Pkg; Pkg.test("VideoIO")'
```

---

**Tempo total de instalação**: 10-15 minutos  
**Dificuldade**: Intermediária  
**Última atualização**: Outubro 2024

🎉 **Boa instalação!**