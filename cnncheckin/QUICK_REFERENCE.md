# 🚀 CNNCheckin - Referência Rápida de Comandos

Guia rápido para uso diário do sistema.

## 📸 Captura de Imagens

### Menu Interativo
```bash
julia cnncheckin_capture.jl
```

### Comandos Diretos

| Comando | Descrição |
|---------|-----------|
| `julia cnncheckin_capture.jl --train "Nome" 15` | Capturar para treinamento inicial (15 fotos) |
| `julia cnncheckin_capture.jl --incremental "Nome" 10` | Capturar para adicionar pessoa (10 fotos) |
| `julia cnncheckin_capture.jl --identify` | Capturar para identificação |
| `julia cnncheckin_capture.jl --cameras` | Listar câmeras disponíveis |
| `julia cnncheckin_capture.jl --preview 0 5` | Preview da câmera 0 por 5 segundos |

---

## 🎓 Treinamento Inicial

### Menu Interativo
```bash
julia cnncheckin_pretrain_webcam.jl
```

### Modo Rápido
```bash
# Capturar e treinar 3 pessoas com 15 fotos cada
julia cnncheckin_pretrain_webcam.jl --quick "Pessoa1" "Pessoa2" "Pessoa3" --num 15

# Treinar sem captura (usar imagens existentes)
julia cnncheckin_pretrain_webcam.jl --no-capture
```

### Parâmetros Padrão
- **Imagens por pessoa**: 15
- **Epochs**: 30 (com early stopping)
- **Batch size**: 8
- **Learning rate**: 0.0001
- **Data augmentation**: Automático

**Tempo estimado**: 5-15 minutos (depende do número de pessoas)

---

## 📚 Aprendizado Incremental

### Menu Interativo
```bash
julia cnncheckin_incremental_webcam.jl
```

### Modo Rápido
```bash
# Adicionar 2 novas pessoas com 10 fotos cada
julia cnncheckin_incremental_webcam.jl --quick "NovaPessoa1" "NovaPessoa2" --num 10

# Treinar incrementalmente sem captura
julia cnncheckin_incremental_webcam.jl --no-capture
```

### Parâmetros Padrão
- **Imagens por pessoa**: 10
- **Epochs**: 15
- **Learning rate**: 0.00005 (mais baixo)
- **Knowledge distillation**: Ativado

**Tempo estimado**: 3-8 minutos por pessoa

---

## 🎯 Identificação

### Menu Interativo
```bash
julia cnncheckin_identify_webcam.jl
```

### Comandos por Modo

#### 1. Identificação Única
```bash
# Identificar quem é a pessoa
julia cnncheckin_identify_webcam.jl --identify

# Usar câmera específica
julia cnncheckin_identify_webcam.jl --identify 1
```

#### 2. Autenticação
```bash
# Verificar se é pessoa específica (threshold 70%)
julia cnncheckin_identify_webcam.jl --auth "Nome Pessoa" 0.7

# Threshold mais rigoroso (80%)
julia cnncheckin_identify_webcam.jl --auth "Nome Pessoa" 0.8
```

#### 3. Modo Contínuo
```bash
# Identificar a cada 5 segundos (ilimitado)
julia cnncheckin_identify_webcam.jl --continuous 5

# Com limite de 20 tentativas
julia cnncheckin_identify_webcam.jl --continuous 10 20
```

#### 4. Check-in/Check-out
```bash
# Sistema de registro de presença
julia cnncheckin_identify_webcam.jl --checkin presenca.csv

# Com arquivo personalizado
julia cnncheckin_identify_webcam.jl --checkin /caminho/arquivo.csv
```

---

## 📊 Consultas e Relatórios

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

# Estatísticas com Julia
julia << 'EOF'
using DelimitedFiles
data = readdlm("presenca.csv", ',', String)
println("Total: ", size(data, 1))
println("Check-ins: ", count(x -> x == "CHECK-IN", data[:, 2]))
println("Check-outs: ", count(x -> x == "CHECK-OUT", data[:, 2]))
EOF
```

### Backup do Modelo
```bash
# Backup completo
tar -czf backup_modelo_$(date +%Y%m%d).tar.gz face_recognition_*.jld2 face_recognition_*.toml

# Backup com data e hora
tar -czf backup_modelo_$(date +%Y%m%d_%H%M%S).tar.gz face_recognition_*

# Restaurar backup
tar -xzf backup_modelo_20241008.tar.gz
```

---

## 🔧 Manutenção

### Retreinar Modelo
```bash
# Com imagens existentes
julia cnncheckin_pretrain_webcam.jl --no-capture
```

### Limpar Cache
```bash
# Remover imagens temporárias
rm -f ../dados/fotos_auth/identificacao_*.jpg
rm -f ../dados/fotos_auth/continuous_*.jpg
rm -f ../dados/fotos_auth/checkin_*.jpg
```

### Verificar Integridade
```bash
# Testar modelo
julia << 'EOF'
using JLD2
try
    data = load("face_recognition_model.jld2")
    println("✅ Modelo OK")
catch e
    println("❌ Erro: ", e)
end
EOF
```

### Atualizar Pacotes
```bash
julia --project -e 'using Pkg; Pkg.update()'
```

---

## 🎨 Personalização

### Ajustar Parâmetros

Edite `cnncheckin_core.jl`:

```julia
# Tamanho da imagem (padrão: 128x128)
const IMG_SIZE = (96, 96)  # Menor = mais rápido

# Batch size (padrão: 8)
const BATCH_SIZE = 4  # Menor = menos RAM

# Epochs (padrão: 30/15)
const PRETRAIN_EPOCHS = 20
const INCREMENTAL_EPOCHS = 10

# Learning rates
const LEARNING_RATE = 0.0001
const INCREMENTAL_LR = 0.00005
```

### Mudar Diretórios

Edite `cnncheckin_core.jl`:

```julia
const TRAIN_DATA_PATH = "/seu/caminho/train"
const INCREMENTAL_DATA_PATH = "/seu/caminho/incremental"
const AUTH_DATA_PATH = "/seu/caminho/auth"
```

---

## 🐛 Troubleshooting Rápido

### Câmera não funciona
```bash
# Listar câmeras
julia cnncheckin_capture.jl --cameras

# Testar preview
julia cnncheckin_capture.jl --preview 0 5

# Linux: verificar permissões
sudo usermod -a -G video $USER
```

### Modelo não carrega
```bash
# Verificar existência
ls -lh face_recognition_model.jld2

# Verificar configuração
cat face_recognition_config.toml

# Retreinar se corrompido
julia cnncheckin_pretrain_webcam.jl --no-capture
```

### Baixa acurácia
```bash
# Adicionar mais fotos por pessoa
julia cnncheckin_capture.jl --train "Nome" 20

# Retreinar
julia cnncheckin_pretrain_webcam.jl --no-capture
```

### Erro de memória
```bash
# Reduzir IMG_SIZE e BATCH_SIZE em cnncheckin_core.jl
# Ou usar menos imagens de treino
```

---

## 📝 Dicas de Uso Diário

### Boas Práticas

✅ **Faça**:
- Backup semanal do modelo
- Mantenha boa iluminação
- Capture mínimo 10 fotos/pessoa
- Varie expressões e ângulos
- Retreine a cada 2-3 meses

❌ **Evite**:
- Capturar com pouca luz
- Usar óculos escuros
- Movimentar durante captura
- Adicionar muitas pessoas de uma vez
- Ignorar avisos de confiança baixa

### Workflow Recomendado

**Setup Inicial** (uma vez):
```bash
1. julia cnncheckin_capture.jl --train "Pessoa1" 15
2. julia cnncheckin_capture.jl --train "Pessoa2" 15
3. julia cnncheckin_pretrain_webcam.jl --no-capture
```

**Adicionar Pessoa** (quando necessário):
```bash
1. julia cnncheckin_capture.jl --incremental "NovaPessoa" 10
2. julia cnncheckin_incremental_webcam.jl --no-capture
```

**Uso Diário**:
```bash
# Sistema de entrada/saída
julia cnncheckin_identify_webcam.jl --checkin presenca_diaria.csv
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

---

## ⚡ Atalhos e Aliases

Adicione ao seu `.bashrc` ou `.zshrc`:

```bash
# Aliases CNNCheckin
alias cnn-capture='cd ~/cnncheckin/src && julia cnncheckin_capture.jl'
alias cnn-train='cd ~/cnncheckin/src && julia cnncheckin_pretrain_webcam.jl'
alias cnn-add='cd ~/cnncheckin/src && julia cnncheckin_incremental_webcam.jl'
alias cnn-identify='cd ~/cnncheckin/src && julia cnncheckin_identify_webcam.jl'
alias cnn-checkin='cd ~/cnncheckin/src && julia cnncheckin_identify_webcam.jl --checkin'
alias cnn-backup='cd ~/cnncheckin/src && tar -czf ../backup_$(date +%Y%m%d).tar.gz face_recognition_*'
```

Uso:
```bash
cnn-capture --train "João" 15
cnn-train --no-capture
cnn-identify --auth "João" 0.7
cnn-checkin presenca.csv
cnn-backup
```

---

## 📞 Comandos de Ajuda

```bash
# Ajuda geral
julia cnncheckin_capture.jl --help
julia cnncheckin_pretrain_webcam.jl --help
julia cnncheckin_incremental_webcam.jl --help
julia cnncheckin_identify_webcam.jl --help

# Versão do Julia
julia --version

# Informações do sistema
julia -e 'using InteractiveUtils; versioninfo()'

# Listar pacotes instalados
julia --project -e 'using Pkg; Pkg.status()'
```

---

**Versão**: 2.0  
**Última atualização**: Outubro 2024  
**Documentação completa**: README_WEBCAM.md

💡 **Dica**: Imprima esta referência para consulta rápida!