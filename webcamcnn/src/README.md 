# 🧠 Sistema CNN de Reconhecimento Facial com Visualização de Camadas

Um sistema completo de reconhecimento facial usando Convolutional Neural Networks (CNN) em Julia, com visualização em tempo real das camadas de treinamento e análise detalhada do processo de aprendizado.

## 📋 Índice

- [Características Principais](#-características-principais)
- [Pré-requisitos](#-pré-requisitos)
- [Instalação](#-instalação)
- [Configuração do Ambiente](#-configuração-do-ambiente)
- [Uso do Sistema](#-uso-do-sistema)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Detalhes Técnicos](#-detalhes-técnicos)
- [Visualizações Geradas](#-visualizações-geradas)
- [Solução de Problemas](#-solução-de-problemas)
- [Contribuição](#-contribuição)

## ✨ Características Principais

- **Captura de Fotos**: Interface para captura de fotos via webcam com múltiplos modos
- **CNN Personalizada**: Rede neural convolucional otimizada para reconhecimento facial
- **Visualização de Camadas**: Visualização em tempo real das ativações de cada camada durante o treinamento
- **Análise Detalhada**: Gráficos e estatísticas completas do processo de aprendizado
- **Interface Intuitiva**: Sistema de menus interativo para fácil navegação
- **Gerenciamento de Dados**: Ferramentas para organizar, limpar e visualizar os dados
- **Exportação de Relatórios**: Geração de relatórios HTML e análises em tempo real

## 🔧 Pré-requisitos

### Software Necessário

- **Julia**: Versão 1.8 ou superior
- **Webcam**: Câmera funcional conectada ao sistema
- **Sistema Operacional**: Windows, Linux ou macOS

### Hardware Recomendado

- **RAM**: Mínimo 8GB, recomendado 16GB
- **Armazenamento**: 2GB de espaço livre
- **GPU**: CUDA compatível (opcional, para aceleração)

## 🚀 Instalação

### 1. Instalar Julia

Baixe e instale Julia do [site oficial](https://julialang.org/downloads/).

### 2. Clonar o Repositório

```bash
git clone https://github.com/seu-usuario/webcamcnn.git
cd webcamcnn
```

### 3. Instalar Dependências Julia

Execute Julia e instale os pacotes necessários:

```julia
using Pkg

# Pacotes principais
Pkg.add(["Flux", "Images", "FileIO", "CUDA", "Statistics", "Random"])
Pkg.add(["JLD2", "TOML", "ImageTransformations", "LinearAlgebra"])
Pkg.add(["Dates", "VideoIO", "Plots", "ColorSchemes"])

# Pacotes de plotting e visualização
Pkg.add(["PlotlyJS", "GR"])

# Para processamento de imagem avançado
Pkg.add(["ImageFiltering", "ImageSegmentation"])
```

### 4. Verificar Instalação da Webcam

Teste se a webcam está funcionando:

```julia
using VideoIO
camera = VideoIO.opencamera()
frame = read(camera)
close(camera)
```

## ⚙️ Configuração do Ambiente

### 1. Estrutura de Diretórios

O sistema criará automaticamente a seguinte estrutura:

```
../../../dados/webcamcnn/
├── photos/           # Fotos capturadas organizadas por pessoa
├── models/           # Modelos treinados e configurações
├── visualizations/   # Visualizações das camadas por pessoa
│   ├── pessoa1/      # Visualizações específicas da pessoa1
│   │   ├── layer_1_Conv_features.png
│   │   ├── layer_2_BatchNorm_features.png
│   │   ├── ...
│   │   └── pessoa1_processing_summary.png
│   ├── pessoa2/      # Visualizações específicas da pessoa2
│   └── training_analysis/  # Análises gerais de treinamento
└── exports/          # Dados exportados
```

### 2. Configurações Padrão

O arquivo `config.jl` contém as seguintes configurações padrão:

```julia
const CONFIG = Dict(
    :img_size => (128, 128),        # Tamanho das imagens
    :batch_size => 8,               # Tamanho do batch
    :epochs => 30,                  # Número de épocas
    :learning_rate => 0.0001,       # Taxa de aprendizado
)
```

## 🎯 Uso do Sistema

### 1. Executar o Sistema

```bash
cd webcamcnn
julia main.jl
```

### 2. Menu Principal

O sistema apresentará o seguinte menu:

```
🎯 MAIN MENU:
1 - 📸 Capture photos from webcam
2 - 🧠 Train face recognition model  
3 - 🔍 Test/predict with trained model
4 - 📊 System information
5 - 🗂️ Manage data (list/clean)
6 - 🎨 Visualization management
7 - ⚙️ Advanced options
8 - 🚪 Exit
```

### 3. Fluxo de Trabalho Recomendado

#### Passo 1: Capturar Fotos (Opção 1)

1. Selecione "Capture photos from webcam"
2. Digite o nome da pessoa
3. Escolha o modo de captura:
   - **Automático**: 10 fotos com intervalo de 3 segundos
   - **Manual**: Controle manual de cada captura
   - **Single com visualização**: Uma foto com análise imediata

**Dicas para melhor captura:**
- Use boa iluminação
- Varie os ângulos (frontal, perfil esquerdo, perfil direito)
- Mantenha expressão neutra
- Evite sombras no rosto

#### Passo 2: Treinar o Modelo (Opção 2)

1. Selecione "Train face recognition model"
2. O sistema verificará se há dados suficientes
3. Confirme o início do treinamento
4. Acompanhe o progresso em tempo real

**O que acontece durante o treinamento:**
- Análise da arquitetura do modelo
- Processamento das imagens com data augmentation
- Treinamento com visualização das camadas
- Criação de gráficos de progresso
- Salvamento automático do melhor modelo

#### Passo 3: Testar o Modelo (Opção 3)

1. Selecione "Test/predict with trained model"
2. Escolha entre:
   - **Webcam ao vivo**: Teste em tempo real
   - **Arquivo de imagem**: Teste com arquivo específico
   - **Teste em lote**: Múltiplas imagens
   - **Monitoramento de confiança**: Análise detalhada

### 4. Gerenciamento de Visualizações (Opção 6)

O sistema oferece várias opções para gerenciar as visualizações:

- **Visualizar existentes**: Ver todas as visualizações criadas
- **Criar para fotos existentes**: Gerar visualizações para fotos já capturadas
- **Análise de treinamento**: Gráficos detalhados do processo de treinamento
- **Exportar galeria**: Criar galeria HTML navegável

## 📂 Estrutura do Projeto

```
webcamcnn/
├── main.jl              # Interface principal do sistema
├── config.jl            # Configurações e funções utilitárias
├── capture.jl           # Módulo de captura de fotos
├── training.jl          # Módulo de treinamento CNN
├── prediction.jl        # Módulo de predição e testes
└── README.md           # Este arquivo
```

### Descrição dos Módulos

**main.jl**
- Interface de usuário principal
- Gerenciamento de menus e fluxos
- Coordenação entre módulos

**config.jl**
- Configurações globais do sistema
- Funções de preprocessamento de imagem
- Criação da arquitetura CNN
- Visualização de camadas

**capture.jl**
- Interface com webcam
- Captura de fotos automática/manual
- Geração de visualizações em tempo real

**training.jl**
- Algoritmos de treinamento
- Análise de performance
- Visualização do progresso
- Salvamento de modelos

**prediction.jl**
- Sistema de predição
- Testes em tempo real
- Análise de confiança
- Comparação de resultados

## 🔍 Detalhes Técnicos

### Arquitetura da CNN

A rede neural implementada segue a seguinte arquitetura:

```
Entrada (128x128x3)
    ↓
Conv2D(3→64, 3x3) + ReLU + BatchNorm
    ↓
MaxPooling(2x2)
    ↓
Conv2D(64→128, 3x3) + ReLU + BatchNorm
    ↓
MaxPooling(2x2)
    ↓
Conv2D(128→256, 3x3) + ReLU + BatchNorm
    ↓
MaxPooling(2x2)
    ↓
Conv2D(256→256, 3x3) + ReLU + BatchNorm
    ↓
MaxPooling(2x2)
    ↓
Flatten
    ↓
Dense(final_features→512) + ReLU + Dropout(0.4)
    ↓
Dense(512→256) + ReLU + Dropout(0.3)
    ↓
Dense(256→num_classes)
    ↓
Saída (probabilidades por classe)
```

### Data Augmentation

O sistema aplica as seguintes técnicas de augmentação:

- **Flip horizontal**: Espelhamento horizontal
- **Variação de brilho**: ±10% de intensidade
- **Ruído gaussiano**: Adição de ruído leve
- **Normalização**: Média zero e desvio padrão unitário

### Processo de Treinamento

1. **Carregamento de dados**: Leitura e preprocessamento das imagens
2. **Divisão train/validation**: 80% treino, 20% validação
3. **Criação de batches**: Agrupamento para processamento eficiente
4. **Treinamento com early stopping**: Parada automática se não houver melhoria
5. **Visualização em tempo real**: Gráficos de loss e accuracy
6. **Análise de camadas**: Monitoramento das ativações

## 🎨 Visualizações Geradas

### Por Pessoa

Para cada pessoa no sistema, são geradas visualizações específicas:

**Estrutura de diretório:**
```
visualizations/
└── [nome_pessoa]/
    ├── layer_1_Conv_features.png         # Mapas de características da 1ª camada
    ├── layer_2_BatchNorm_activations.png # Ativações de normalização
    ├── layer_3_MaxPool_features.png      # Resultado do max pooling
    ├── ...                               # Demais camadas
    ├── [pessoa]_processing_summary.png   # Resumo completo do processamento
    └── [pessoa]_layer_info.txt          # Informações detalhadas em texto
```

### Tipos de Visualização

1. **Feature Maps Convolucionais**
   - Visualização em grid dos filtros ativados
   - Cores representam intensidades de ativação
   - Cada canal mostrado separadamente

2. **Ativações de Camadas Dense**
   - Gráficos de barras das ativações dos neurônios
   - Distribuição das ativações
   - Análise estatística

3. **Resumo do Processamento**
   - Gráficos de magnitude de ativação por camada
   - Evolução do tamanho dos feature maps
   - Distribuição da camada final
   - Composição dos tipos de camadas

4. **Análise de Treinamento**
   - Curvas de loss e accuracy
   - Evolução dos pesos durante o treinamento
   - Estatísticas por época
   - Análise de convergência

### Análises Especiais

**Decision Analysis**
- Mapas de atenção mostrando quais regiões da imagem mais contribuem para a decisão
- Análise da importância de cada camada
- Distribuição de probabilidades entre classes

**Comparison Analysis**
- Comparação de predições entre diferentes fotos
- Heatmaps de probabilidades
- Análise de consistência

## 🔧 Solução de Problemas

### Problemas Comuns

**1. Erro: "Camera not found"**
```
Solução:
- Verifique se a webcam está conectada
- Feche outros programas que usam a câmera
- Teste com: VideoIO.get_camera_devices()
```

**2. Erro: "LoadError: ArgumentError: Package X not found"**
```
Solução:
- Execute: using Pkg; Pkg.add("X")
- Reinicie Julia
- Verifique a versão do Julia (>= 1.8)
```

**3. Erro: "Out of memory during training"**
```
Solução:
- Reduza batch_size no config.jl
- Reduza img_size para (64, 64)
- Feche outros programas
```

**4. Baixa accuracy no treinamento**
```
Solução:
- Capture mais fotos por pessoa (mínimo 10)
- Melhore a qualidade/iluminação das fotos
- Aumente o número de epochs
- Varie mais os ângulos das fotos
```

**5. Visualizações não aparecem**
```
Solução:
- Verifique se Plots.jl está instalado
- Teste: using Plots; plot([1,2,3])
- Instale backend: Pkg.add("GR") ou Pkg.add("PlotlyJS")
```

### Logs de Debug

Para ativar logs detalhados, modifique o arquivo `config.jl`:

```julia
const DEBUG_MODE = true  # Adicione esta linha no início

function debug_log(message)
    if DEBUG_MODE
        println("[DEBUG $(Dates.now())]: $message")
    end
end
```

### Otimização de Performance

**Para sistemas com GPU CUDA:**
```julia
using CUDA
model = model |> gpu  # Mover modelo para GPU
```

**Para sistemas com pouca memória:**
```julia
# No config.jl, ajustar:
:batch_size => 4,          # Reduzir batch size
:img_size => (64, 64),     # Reduzir tamanho da imagem
```

## 🚀 Funcionalidades Avançadas

### Exportação de Dados

O sistema permite exportar:

- **Modelos treinados**: Em formato JLD2 para reutilização
- **Configurações**: Arquivos TOML com metadados completos
- **Visualizações**: Galeria HTML navegável
- **Relatórios**: Análises detalhadas em texto e gráficos

### Análise de Arquitetura

Através da opção "Advanced > Model architecture analysis":

- **Contagem de parâmetros**: Total e por tipo de camada
- **Tamanho do modelo**: Estimativa em MB
- **Análise de complexidade**: Operações por inferência
- **Visualização da arquitetura**: Diagrama da rede

### Backup e Restauração

Sistema completo de backup:

- **Backup automático**: Antes de operações críticas
- **Backup manual**: Através do menu avançado
- **Restauração**: Recuperação de estados anteriores
- **Versionamento**: Controle de múltiplas versões

## 🤝 Contribuição

Para contribuir com o projeto:

1. Fork o repositório
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

### Guidelines de Desenvolvimento

- **Código**: Siga as convenções de Julia
- **Documentação**: Documente todas as funções principais
- **Testes**: Inclua testes para novas funcionalidades
- **Compatibilidade**: Mantenha compatibilidade com Julia 1.8+

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE.md](LICENSE.md) para detalhes.

## 📞 Suporte

Para suporte ou dúvidas:

- **Issues**: Abra uma issue no GitHub
- **Documentação**: Consulte este README
- **Comunidade Julia**: [JuliaLang Discourse](https://discourse.julialang.org/)

## 📚 Recursos Adicionais

### Tutoriais Recomendados

1. **Julia para Machine Learning**: [MLJ.jl Tutorial](https://alan-turing-institute.github.io/MLJ.jl/dev/)
2. **Flux.jl Documentation**: [Oficial Flux.jl](https://fluxml.ai/Flux.jl/stable/)
3. **Computer Vision com Julia**: [JuliaImages](https://juliaimages.org/latest/)

### Papers e Referências

- **CNNs para Reconhecimento Facial**: LeCun et al., 1998
- **Batch Normalization**: Ioffe & Szegedy, 2015
- **Data Augmentation**: Shorten & Khoshgoftaar, 2019

---

**Sistema desenvolvido com ❤️ em Julia**

*Versão 4.0-Enhanced-LayerViz*