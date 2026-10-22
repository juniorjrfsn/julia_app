# 🎥 CNNCheckin - Sistema de Reconhecimento Facial

O projeto foi reorganizado para um único ponto de entrada principal em `src/cnncheckin.jl`, com utilitários específicos como `src/capture_iriun.jl` para câmeras Iriun.

## Estrutura atual

```text
cnncheckin/
├── Project.toml
├── README.md
├── src/
│   ├── cnncheckin.jl       # núcleo principal
│   ├── capture_iriun.jl    # captura específica para webcam via Iriun
│   ├── face_detect.py      # detecção de rosto auxiliar
│   ├── face_recognition_model.jld2
│   ├── face_recognition_config.toml
│   └── face_recognition_model_data.toml
├── dados/
│   ├── fotos_train/
│   ├── fotos_new/
│   └── fotos_auth/
└── ...
```

---

## Início rápido

```bash
cd d:/projetos/julia_app/cnncheckin

# instalar dependências do projeto
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# listar câmeras disponíveis
julia --project=. src/cnncheckin.jl --list-cameras

# treinar com uma pasta de imagens
julia --project=. src/cnncheckin.jl --train dados/fotos_train

# identificar uma imagem
julia --project=. src/cnncheckin.jl --identify dados/fotos_auth/exemplo.jpg
```

---

## Comandos principais

### 1) Listar câmeras

```bash
julia --project=. src/cnncheckin.jl --list-cameras
```

### 2) Capturar imagens para treinamento

```bash
# captura de uma pessoa em uma pasta
julia --project=. src/cnncheckin.jl --capture "João" dados/fotos_train 15
```

Parâmetros:
- `pessoa`: nome da pessoa
- `pasta`: diretório de destino
- `num`: quantidade de imagens (opcional, padrão: 10)

### 3) Treinar o modelo

```bash
julia --project=. src/cnncheckin.jl --train dados/fotos_train
```

### 4) Identificar pessoa em imagem

```bash
julia --project=. src/cnncheckin.jl --identify dados/fotos_auth/joao.jpg
```

### 5) Ajuda

```bash
julia --project=. src/cnncheckin.jl --help
```

---

## Captura com Iriun Webcam

Se a câmera for conectada via Iriun, use o script específico:

```bash
cd d:/projetos/julia_app/cnncheckin/src
julia capture_iriun.jl --list
julia capture_iriun.jl --single foto_teste.jpg --camera 0
julia capture_iriun.jl --multiple "João" ./fotos 15 --camera 0
```

Comandos suportados no script de captura:
- `--list`
- `--test`
- `--single <arquivo.jpg> [--camera N]`
- `--multiple <nome> <pasta> <num> [--camera N]`
- `--help`

---

## Organização recomendada de dados

Crie as pastas antes do treinamento:

```bash
mkdir -p dados/fotos_train
mkdir -p dados/fotos_new
mkdir -p dados/fotos_auth
```

Exemplo de nomes de arquivo esperados:

```text
dados/fotos_train/joao_01.jpg
dados/fotos_train/joao_02.jpg
dados/fotos_train/maria_01.jpg
```

O nome da pessoa é extraído a partir do arquivo, então o padrão simples ajuda a manter a identificação correta.

---

## Requisitos

- Julia 1.10+
- Pacotes no `Project.toml`:
  - `Flux`
  - `Images`
  - `FileIO`
  - `VideoIO`
  - `JLD2`
  - `TOML`
  - `ImageTransformations`

Caso queira instalar manualmente:

```bash
julia --project=. -e 'using Pkg; Pkg.add(["Flux","Images","FileIO","VideoIO","JLD2","TOML","ImageTransformations"])'
```

---

## Observações

- O projeto agora usa um único módulo principal, o que facilita manutenção e execução.
- Os scripts antigos foram substituídos por uma estrutura mais enxuta, evitando duplicação e bugs de integração.
- O arquivo `capture_iriun.jl` continua disponível para cenários específicos de webcam via celular ou Iriun.

---

## Fluxo recomendado

```bash
# 1. capturar imagens de cada pessoa
julia --project=. src/cnncheckin.jl --capture "João" dados/fotos_train 15
julia --project=. src/cnncheckin.jl --capture "Maria" dados/fotos_train 15

# 2. treinar
julia --project=. src/cnncheckin.jl --train dados/fotos_train

# 3. validar/identificar
julia --project=. src/cnncheckin.jl --identify dados/fotos_auth/joao.jpg
```
