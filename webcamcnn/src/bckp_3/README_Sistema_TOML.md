# 🤖 CNN Checkin System v2.0 - Sistema com Pesos e Vieses em TOML

Sistema avançado de reconhecimento facial que salva **todos os pesos e vieses** dos treinamentos em formato **TOML**, permitindo acúmulo de múltiplos treinamentos e análise comparativa.

## 🆕 Principais Novidades

### ✅ **Formato TOML para Pesos e Vieses**
- Todos os parâmetros do modelo são salvos em formato TOML legível
- Suporte para acúmulo de múltiplos treinamentos no mesmo arquivo
- Fácil adição de novos registros sem perder histórico

### ✅ **Gestão Completa de Treinamentos**
- Múltiplos treinamentos salvos em um único arquivo
- Comparação entre diferentes versões
- Análise de evolução da performance
- Sistema de backup e exportação

### ✅ **Compatibilidade Mantida**
- Sistema antigo JLD2 mantido para compatibilidade
- Migração gradual para TOML
- Ambos os formatos coexistem

## 📁 Estrutura de Arquivos

```
projeto/
├── src/
│   ├── core.jl                    # Módulo principal (mantido)
│   ├── capture_and_train.jl       # Captura de webcam (mantido)
│   ├── pretrain.jl               # Versão original (mantido)
│   ├── weights_manager.jl        # 🆕 Gerenciador de pesos TOML
│   ├── pretrain_modified.jl      # 🆕 Versão com suporte TOML
│   ├── weights_utils.jl          # 🆕 Utilitários de gestão
│   └── main_toml_system.jl       # 🆕 Sistema principal integrado
├── model_weights.toml            # 🆕 Pesos acumulados de todos treinamentos
├── face_recognition_config.toml  # Configuração (mantido)
├── face_recognition_model.jld2   # Modelo em JLD2 (mantido)
└── fotos_rosto/                  # Dados de treino (mantido)
```

## 🚀 Como Usar

### 1. **Execução Simples**
```bash
julia main_toml_system.jl
```

### 2. **Menu Principal**
```
🎯 MENU PRINCIPAL:
1️⃣  - Capturar fotos e treinar novo modelo
2️⃣  - Gerenciar treinamentos (listar, comparar, carregar)
3️⃣  - Utilitários de pesos TOML (backup, análise, limpeza)
4️⃣  - Executar apenas novo treinamento (dados existentes)
5️⃣  - Mostrar detalhes do sistema
6️⃣  - Sair
```

### 3. **Captura e Treinamento**
- Sistema integrado de captura via webcam
- Treinamento automático com salvamento em TOML
- Acúmulo automático de novos treinamentos

## 📊 Formato TOML dos Pesos

### Estrutura Geral
```toml
[format_info]
version = "1.0"
description = "CNN Face Recognition Weights and Biases"
created_by = "webcamcnn.jl"
last_updated = "2025-01-15T10:30:45"

[summary]
total_trainings = 3
latest_training = "train_20250115_103045_a1b2c3d4"
all_persons = ["joao", "maria", "pedro"]

[trainings.train_ID_UNICO]
# Cada treinamento tem um ID único baseado em timestamp + hash
```

### Metadados de Treinamento
```toml
[trainings.train_ID.metadata]
training_id = "train_20250115_103045_a1b2c3d4"
timestamp = "2025-01-15T10:30:45"
person_names = ["joao", "maria", "pedro"]
epochs_trained = 25
final_accuracy = 0.892
best_epoch = 22
model_architecture = "CNN_FaceRecognition_v1"
learning_rate = 0.0001
batch_size = 8
data_hash = "a1b2c3d4e5f6g7h8"
```

### Pesos das Camadas
```toml
[trainings.train_ID.layers.layer_1]
layer_type = "Conv{2, 4, typeof(relu), ...}"
layer_index = 1

[trainings.train_ID.layers.layer_1.weights]
shape = [3, 3, 3, 64]  # kernel_h, kernel_w, in_channels, out_channels
dtype = "Float32"
values = [[...], [...], ...]  # Dados organizados
stats = { mean = 0.0012, std = 0.1856, min = -0.4123, max = 0.3987, count = 1728 }

[trainings.train_ID.layers.layer_1.bias]
shape = [64]
dtype = "Float32"
values = [0.0123, -0.0045, ...]
stats = { mean = 0.0001, std = 0.0234, min = -0.0567, max = 0.0456, count = 64 }
```

### Camadas BatchNorm
```toml
[trainings.train_ID.layers.layer_2.beta]    # Bias learnable
[trainings.train_ID.layers.layer_2.gamma]   # Scale learnable
[trainings.train_ID.layers.layer_2.mu]      # Running mean
[trainings.train_ID.layers.layer_2.sigma_squared]  # Running variance
```

## 🛠️ Funcionalidades Principais

### 1. **Gerenciamento de Treinamentos**
```julia
# Listar todos os treinamentos
list_saved_trainings("../../../dados/webcamcnn/model_weights.toml")

# Comparar dois treinamentos
compare_training_weights("../../../dados/webcamcnn/model_weights.toml", "train_id1", "train_id2")

# Carregar treinamento específico
training_data = load_weights_from_toml("../../../dados/webcamcnn/model_weights.toml", "train_id")
```

### 2. **Análise e Estatísticas**
```julia
# Análise de evolução completa
analyze_training_evolution("../../../dados/webcamcnn/model_weights.toml")

# Validar integridade do arquivo
validate_weights_file("../../../dados/webcamcnn/model_weights.toml")
```

### 3. **Backup e Exportação**
```julia
# Criar backup automático
backup_weights("../../../dados/webcamcnn/model_weights.toml")

# Exportar treinamento específico
export_training("../../../dados/webcamcnn/model_weights.toml", "train_id", "../../../dados/webcamcnn/exported_model.toml")

# Importar de outro sistema
import_training("../../../dados/webcamcnn/external_model.toml", "../../../dados/webcamcnn/model_weights.toml")
```

### 4. **Limpeza e Manutenção**
```julia
# Manter apenas os 5 treinamentos mais recentes
cleanup_old_trainings("../../../dados/webcamcnn/model_weights.toml", 5)
```

## 📈 Vantagens do Sistema TOML

### **1. Legibilidade Humana**
- Formato texto puro, fácil de ler e entender
- Estrutura hierárquica clara
- Comentários e documentação inline

### **2. Acúmulo de Dados**
- Múltiplos treinamentos no mesmo arquivo
- Histórico completo preservado
- Fácil adição de novos treinamentos

### **3. Análise Comparativa**
- Compare performance entre versões
- Identifique melhor época de treinamento
- Analise evolução dos pesos

### **4. Portabilidade**
- Padrão universal (TOML)
- Compatível com qualquer linguagem
- Fácil integração com sistemas externos

### **5. Debugging Avançado**
- Examine pesos específicos manualmente
- Identifique camadas problemáticas
- Análise estatística detalhada

## 🔧 Configurações Avançadas

### Constantes Personalizáveis
```julia
# Em core.jl
const IMG_SIZE = (128, 128)          # Tamanho das imagens
const BATCH_SIZE = 8                 # Tamanho do batch
const PRETRAIN_EPOCHS = 30           # Épocas de pré-treino
const LEARNING_RATE = 0.0001         # Taxa de aprendizagem

# Em weights_manager.jl
const WEIGHTS_TOML_PATH = "../../../dados/webcamcnn/model_weights.toml"  # Arquivo de pesos
```

### Estrutura de IDs Únicos
```julia
# Formato: train_YYYYMMDD_HHMMSS_HASH
# Exemplo: train_20250115_103045_a1b2c3d4
training_id = generate_training_id(person_names)
```

## 📊 Exemplo de Uso Completo

### 1. **Primeiro Treinamento**
```bash
# Executar sistema
julia main_toml_system.jl

# Escolher opção 1 (Capturar e treinar)
# Sistema captura fotos e treina automaticamente
# Pesos salvos em model_weights.toml
```

### 2. **Adicionar Nova Pessoa**
```bash
# Adicionar fotos da nova pessoa na pasta fotos_rosto/
# Executar novo treinamento (opção 4)
# Sistema detecta mudanças e treina modelo expandido
# Novo treinamento adicionado ao arquivo TOML
```

### 3. **Análise de Performance**
```bash
# Opção 2 -> Comparar treinamentos
# Sistema mostra evolução da acurácia
# Identifica melhor modelo automaticamente
```

## 🧮 Detalhes Técnicos

### **Conversão de Arrays**
- **1D**: Salvos diretamente como arrays TOML
- **2D**: Convertidos para array de arrays
- **3D/4D**: Estrutura especial com metadados de shape
- **Reconstrução**: Função automática para restaurar dimensões

### **Estatísticas por Camada**
- **Média, desvio padrão, min, max**
- **Contagem de parâmetros**
- **Análise de distribuição**

### **Otimizações de Armazenamento**
- **Compressão inteligente** para grandes tensores
- **Metadados separados** dos dados brutos
- **Índices hierárquicos** para acesso rápido

## 🔍 Debugging e Monitoramento

### **Logs Detalhados**
```julia
# Sistema mostra progresso detalhado:
📂 Carregando dados de pré-treino...
✅ Carregado: joao_1_2025-01-15_10-30-22.jpg -> joao (5 variações)
👤 Pessoa: joao - 15 imagens (Label: 1)
🗃️ Criando modelo CNN...
📈 Epoch 25/30 - Loss: 0.0234 - Val Acc: 89.2% - Best: 89.2% (Epoch 25)
💾 Salvando pesos e vieses em formato TOML...
✅ Pesos salvos em: model_weights.toml
```

### **Validação Automática**
- Verifica integridade dos arquivos
- Detecta corrupção de dados
- Valida estrutura TOML
- Confirma compatibilidade de versões

## ⚡ Performance e Limites

### **Capacidade**
- **Treinamentos simultâneos**: Ilimitado
- **Tamanho por modelo**: ~5-20 MB
- **Pessoas por modelo**: Testado até 50+
- **Épocas por treinamento**: 1-100+

### **Otimizações**
- Carregamento sob demanda
- Cache inteligente
- Compressão de dados redundantes
- Índices de busca rápida

## 🔧 Solução de Problemas

### **Erro: Arquivo TOML Corrompido**
```julia
# Usar backup automático
backup_weights("../../../dados/webcamcnn/model_weights.toml")
# Sistema cria backup_TIMESTAMP.toml

# Validar integridade
validate_weights_file("../../../dados/webcamcnn/model_weights.toml")
```

### **Erro: Treinamento Não Converge**
```julia
# Verificar dados de entrada
verificar_dados_treino()

# Ajustar hiperparâmetros em core.jl
const LEARNING_RATE = 0.00005  # Reduzir learning rate
const PRETRAIN_EPOCHS = 50     # Aumentar épocas
```

### **Erro: Memória Insuficiente**
```julia
# Reduzir batch size
const BATCH_SIZE = 4

# Reduzir tamanho da imagem
const IMG_SIZE = (96, 96)
```

## 📚 Integração com Sistemas Externos

### **Python Integration**
```python
import toml

# Carregar pesos do Julia
weights = toml.load('model_weights.toml')
training = weights['trainings']['train_20250115_103045_a1b2c3d4']

# Acessar pesos específicos
layer1_weights = training['layers']['layer_1']['weights']['values']
accuracy = training['metadata']['final_accuracy']
```

### **Export para TensorFlow/PyTorch**
```julia
# Função customizada para exportar
function export_to_tensorflow(training_id::String)
    # Carregar pesos TOML
    # Converter para formato TF/PyTorch
    # Salvar em formato compatível
end
```

## 🎯 Roadmap Futuro

### **v2.1 - Planejado**
- [ ] Suporte a modelos de diferentes arquiteturas
- [ ] Exportação automática para ONNX
- [ ] Interface web para visualização
- [ ] Treinamento distribuído

### **v2.2 - Planejado**
- [ ] Versionamento automático de modelos
- [ ] A/B testing de diferentes versões
- [ ] Métricas avançadas de performance
- [ ] Integração com MLflow

## 📖 Referências

- **TOML Specification**: https://toml.io/
- **Flux.jl Documentation**: https://fluxml.ai/
- **Julia Pkg Manager**: https://julialang.org/

## 🤝 Contribuição

Sistema desenvolvido para maximizar flexibilidade e extensibilidade. Contribuições são bem-vindas!

### **Como Contribuir**
1. Fork o projeto
2. Crie branch para feature (`git checkout -b feature/nova-funcionalidade`)
3. Commit mudanças (`git commit -am 'Adiciona nova funcionalidade'`)
4. Push para branch (`git push origin feature/nova-funcionalidade`)
5. Abra Pull Request

---

## 🏆 **CNN Checkin System v2.0 - Revolução no Gerenciamento de Modelos de Deep Learning!**


Perfeito! Criei um sistema completo de visualização de camadas para seu projeto CNN. Aqui está o que foi implementado:
Sistema de Visualização de Camadas
1. layer_visualization.jl - Core do sistema

Extração de ativações: Captura saídas intermediárias de cada camada
Visualização de filtros: Mostra padrões que cada filtro convoluciональ detecta
Organização por pessoa: Cria estrutura hierárquica pessoa/layer_N/
Normalização automática: Converte ativações em imagens interpretáveis

2. pretrain_modified_with_visualization.jl - Integração com treino

Salvamento automático: Gera visualizações durante épocas específicas
Relatórios detalhados: Documenta cada sessão de treino
Controle de frequência: Configurável (padrão: a cada 5 épocas)

3. integration_with_visualizations.jl - Setup do sistema

Verificação de dependências: Instala pacotes necessários
Configuração automática: Cria diretórios e arquivos de config
Sistema de testes: Valida funcionamento

Estrutura de Arquivos Gerada
dados/webcamcnn/layer_visualizations/
├── training_20250105_143022/
│   ├── João/
│   │   ├── layer_1/
│   │   │   ├── sample_1_layer1_activation.png
│   │   │   ├── sample_1_layer1_filter1.png
│   │   │   └── sample_1_layer1_filter2.png
│   │   ├── layer_2/
│   │   └── layer_3/
│   ├── Maria/
│   │   ├── layer_1/
│   │   └── layer_2/
│   └── README.md
└── training_report_20250105_143022.md
Como Usar
1. Instalação
bashjulia integration_with_visualizations.jl
2. Execução com Visualizações
No main_toml_system.jl, adicione:
juliainclude("layer_visualization.jl")
include("pretrain_modified_with_visualization.jl")
3. Menu Principal
Adicione opção 7 no menu:
juliaelseif choice == "7"
    visualization_management_menu()
Benefícios do Sistema
Análise de Aprendizado

Camadas 1-2: Detectores de bordas e texturas básicas
Camadas 3-4: Combinações complexas (olhos, nariz, boca)
Camadas finais: Características específicas por pessoa

Detecção de Problemas

Filtros mortos: Todos zeros (não aprendeu)
Redundância: Filtros muito similares
Overfitting: Ativações muito específicas

Otimização

Compare épocas diferentes para ver evolução
Identifique gargalos de aprendizado
Ajuste hiperparâmetros baseado nas visualizações

Implementação Técnica
O sistema salva automaticamente:

Ativações: Como cada imagem é "vista" pela rede
Filtros: Padrões que cada neurônio detecta
Estatísticas: Resumos numéricos para análise
Relatórios: Documentação completa de cada treino

Essa abordagem permite mapear exatamente o que sua rede neural está aprendendo, facilitando debugging, otimização e compreensão do modelo.