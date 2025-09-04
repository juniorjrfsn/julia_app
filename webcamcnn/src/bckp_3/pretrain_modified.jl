# projeto: webcamcnn
# file: webcamcnn/src/pretrain_modified.jl


# projeto: webcamcnn
# file: webcamcnn/src/pretrain_modified_with_visualization.jl

# Versão modificada do pretrain_modified.jl com suporte a visualizações de camadas

using Flux
using Statistics
using Random
using JLD2
using Dates

include("core.jl")
include("weights_manager.jl")
include("layer_visualization.jl")  # Nova funcionalidade
using .CNNCheckinCore

# Constante para o arquivo de pesos
const WEIGHTS_TOML_PATH = "model_weights.toml"

# Pre-training function com visualizações integradas
function pretrain_model_with_visualizations(model, train_data, val_data, epochs, learning_rate, 
                                          person_names::Vector{String}; 
                                          save_viz_every::Int = 5)
    println("🚀 Iniciando pré-treino com visualizações de camadas...")
    
    optimizer = ADAM(learning_rate, (0.9, 0.999), 1e-8)
    opt_state = Flux.setup(optimizer, model)
    
    train_losses = Float64[]
    val_accuracies = Float64[]
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    patience_limit = 10
    
    visualization_paths = String[]  # Armazenar caminhos das visualizações
    
    for epoch in 1:epochs
        epoch_loss = 0.0
        num_batches = 0
        
        println("\n📊 Época $epoch/$epochs")
        
        # Training phase
        for (x, y) in train_data
            try
                loss, grads = Flux.withgradient(model) do m
                    ŷ = m(x)
                    Flux.logitcrossentropy(ŷ, y)
                end
                Flux.update!(opt_state, model, grads[1])
                epoch_loss += loss
                num_batches += 1
            catch e
                println("❌ Erro no treinamento do batch: $e")
                continue
            end
        end
        
        avg_loss = epoch_loss / num_batches
        push!(train_losses, avg_loss)
        
        # Validation phase
        val_acc = pretrain_accuracy(model, val_data)
        push!(val_accuracies, val_acc)
        
        println("Epoch $epoch/$epochs - Loss: $(round(avg_loss, digits=6)) - Val Acc: $(round(val_acc*100, digits=2))%")
        
        # Salvar visualizações em épocas específicas
        if epoch % save_viz_every == 0 || epoch == 1 || epoch == epochs
            println("🎨 Gerando visualizações para época $epoch...")
            viz_path = add_visualization_to_training(model, train_data, person_names, epoch, 1)
            if viz_path !== nothing
                push!(visualization_paths, viz_path)
                println("✅ Visualizações salvas em: $(basename(viz_path))")
            end
        end
        
        # Early stopping
        if val_acc > best_val_acc
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            println("🏆 Nova melhor acurácia: $(round(best_val_acc*100, digits=2))%")
        else
            patience_counter += 1
            if patience_counter >= patience_limit
                println("⏹️ Early stopping na época $epoch (paciência esgotada)")
                break
            end
        end
        
        # Mostrar progresso
        progress = epoch / epochs * 100
        println("📈 Progresso: $(round(progress, digits=1))% - Melhor época: $best_epoch")
    end
    
    # Gerar visualização final se não foi gerada na última época
    if epochs % save_viz_every != 0
        println("🎨 Gerando visualizações finais...")
        final_viz_path = generate_training_visualizations(model, train_data, person_names)
        push!(visualization_paths, final_viz_path)
    end
    
    return train_losses, val_accuracies, best_val_acc, best_epoch, visualization_paths
end

# Calculate accuracy (mesmo que antes)
function pretrain_accuracy(model, data_loader)
    correct = 0
    total = 0
    for (x, y) in data_loader
        try
            ŷ = softmax(model(x))
            pred = Flux.onecold(ŷ)
            true_labels = Flux.onecold(y)
            correct += sum(pred .== true_labels)
            total += length(true_labels)
        catch e
            println("❌ Erro calculando acurácia: $e")
            continue
        end
    end
    return total > 0 ? correct / total : 0.0
end

# Load initial training data (mesmo que antes)
function load_pretrain_data(data_path::String; use_augmentation::Bool = true)
    println("📂 Carregando dados de pré-treino...")
    
    if !isdir(data_path)
        error("Diretório $data_path não encontrado!")
    end
    
    person_images = Dict{String, Vector{Array{Float32, 3}}}()
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    
    for filename in readdir(data_path)
        file_ext = lowercase(splitext(filename)[2])
        if file_ext in image_extensions
            img_path = joinpath(data_path, filename)
            if !CNNCheckinCore.validate_image_file(img_path)
                continue
            end
            
            person_name = CNNCheckinCore.extract_person_name(filename)
            img_arrays = CNNCheckinCore.preprocess_image(img_path; augment=use_augmentation)
            
            if img_arrays !== nothing
                if !haskey(person_images, person_name)
                    person_images[person_name] = Vector{Array{Float32, 3}}()
                end
                for img_array in img_arrays
                    push!(person_images[person_name], img_array)
                end
                total_imgs = use_augmentation ? length(img_arrays) : 1
                println("✅ Carregado: $filename -> $person_name ($total_imgs variações)")
            else
                println("❌ Falha ao carregar: $filename")
            end
        end
    end
    
    people_data = Vector{CNNCheckinCore.PersonData}()
    person_names = sort(collect(keys(person_images)))
    
    # Create person data with proper indexing
    for (idx, person_name) in enumerate(person_names)
        images = person_images[person_name]
        if length(images) > 0
            push!(people_data, CNNCheckinCore.PersonData(person_name, images, idx, false))
            println("👤 Pessoa: $person_name - $(length(images)) imagens (Label: $idx)")
        end
    end
    
    return people_data, person_names
end

# Create balanced datasets (mesmo que antes)
function create_pretrain_datasets(people_data::Vector{CNNCheckinCore.PersonData}, split_ratio::Float64 = 0.8)
    println("🔄 Criando datasets de treino e validação...")
    
    train_images = Vector{Array{Float32, 3}}()
    train_labels = Vector{Int}()
    val_images = Vector{Array{Float32, 3}}()
    val_labels = Vector{Int}()
    
    for person in people_data
        n_imgs = length(person.images)
        n_train = max(1, Int(floor(n_imgs * split_ratio)))
        indices = randperm(n_imgs)
        
        for i in 1:n_train
            push!(train_images, person.images[indices[i]])
            push!(train_labels, person.label)
        end
        
        for i in (n_train+1):n_imgs
            push!(val_images, person.images[indices[i]])
            push!(val_labels, person.label)
        end
        
        println("   - $(person.name): $n_train treino, $(n_imgs - n_train) validação")
    end
    
    println("📊 Dataset criado:")
    println("   - Treino: $(length(train_images)) imagens")
    println("   - Validação: $(length(val_images)) imagens")
    
    return (train_images, train_labels), (val_images, val_labels)
end

# Create training batches (mesmo que antes)
function create_pretrain_batches(images, labels, batch_size)
    batches = []
    n_samples = length(images)
    if n_samples == 0
        return batches
    end
    
    unique_labels = unique(labels)
    min_label = minimum(unique_labels)
    max_label = maximum(unique_labels)
    label_range = min_label:max_label
    
    println("🏷️ Labels únicos: $unique_labels")
    println("🏷️ Range de labels: $label_range")
    
    for i in 1:batch_size:n_samples
        end_idx = min(i + batch_size - 1, n_samples)
        batch_images = images[i:end_idx]
        batch_labels = labels[i:end_idx]
        batch_tensor = cat(batch_images..., dims=4)
        
        try
            batch_labels_onehot = Flux.onehotbatch(batch_labels, label_range)
            push!(batches, (batch_tensor, batch_labels_onehot))
            println("   📦 Batch $(div(i-1, batch_size)+1): $(length(batch_labels)) amostras")
        catch e
            println("❌ Erro criando batch $i-$end_idx: $e")
            continue
        end
    end
    
    return batches
end

# CNN architecture (mesmo que antes)
function create_pretrain_cnn_model(num_classes::Int, input_size::Tuple{Int, Int} = CNNCheckinCore.IMG_SIZE)
    # Calculate final feature size after convolutions and pooling
    final_size = div(div(div(div(input_size[1], 2), 2), 2), 2)
    final_features = 256 * final_size * final_size
    
    return Chain(
        # Feature extraction layers
        Conv((3, 3), 3 => 64, relu, pad=1),
        BatchNorm(64),
        Dropout(0.1),
        MaxPool((2, 2)),
        
        Conv((3, 3), 64 => 128, relu, pad=1),
        BatchNorm(128),
        Dropout(0.1),
        MaxPool((2, 2)),
        
        Conv((3, 3), 128 => 256, relu, pad=1),
        BatchNorm(256),
        Dropout(0.15),
        MaxPool((2, 2)),
        
        Conv((3, 3), 256 => 256, relu, pad=1),
        BatchNorm(256),
        Dropout(0.15),
        MaxPool((2, 2)),
        
        # Classifier layers
        Flux.flatten,
        Dense(final_features, 512, relu),
        Dropout(0.4),
        Dense(512, 256, relu),
        Dropout(0.3),
        Dense(256, num_classes)
    )
end

# Main pre-training command com visualizações
function pretrain_command_with_visualizations(save_viz_every::Int = 5)
    println("\n🧠 EXECUTANDO PRÉ-TREINAMENTO COM VISUALIZAÇÕES DE CAMADAS")
    println("=" ^ 60)
    
    start_time = time()
    
    try
        # Load pre-training data
        people_data, person_names = load_pretrain_data(CNNCheckinCore.TRAIN_DATA_PATH; use_augmentation=true)
        if length(people_data) == 0
            error("Nenhum dado de treino encontrado!")
        end
        
        num_classes = length(person_names)
        println("👥 Total de pessoas: $num_classes")
        total_images = sum(length(person.images) for person in people_data)
        println("📸 Total de imagens (com augmentação): $total_images")
        
        # Preparar estrutura de visualizações
        println("🎨 Preparando sistema de visualizações...")
        create_visualization_directories(VIZ_OUTPUT_PATH, person_names)
        
        # Verificar se já existem treinamentos anteriores
        if isfile(WEIGHTS_TOML_PATH)
            println("\n📚 Treinamentos anteriores encontrados:")
            list_saved_trainings(WEIGHTS_TOML_PATH)
            println()
        else
            println("\n🆕 Primeiro treinamento - criando arquivo de pesos")
        end
        
        # Create datasets and batches
        (train_images, train_labels), (val_images, val_labels) = create_pretrain_datasets(people_data)
        train_batches = create_pretrain_batches(train_images, train_labels, CNNCheckinCore.BATCH_SIZE)
        val_batches = create_pretrain_batches(val_images, val_labels, CNNCheckinCore.BATCH_SIZE)
        
        if length(train_batches) == 0
            error("Não foi possível criar batches de treino!")
        end
        
        # Create and train model
        println("🗃️ Criando modelo CNN...")
        model = create_pretrain_cnn_model(num_classes)
        
        # Executar treinamento com visualizações
        train_losses, val_accuracies, best_val_acc, best_epoch, viz_paths = pretrain_model_with_visualizations(
            model, train_batches, val_batches, 
            CNNCheckinCore.PRETRAIN_EPOCHS, 
            CNNCheckinCore.LEARNING_RATE,
            person_names;
            save_viz_every = save_viz_every
        )
        
        end_time = time()
        duration_minutes = (end_time - start_time) / 60
        
        # Prepare training info
        training_info = Dict(
            "epochs_trained" => length(val_accuracies),
            "final_accuracy" => best_val_acc,
            "best_epoch" => best_epoch,
            "total_training_images" => length(train_images),
            "total_validation_images" => length(val_images),
            "augmentation_used" => true,
            "duration_minutes" => duration_minutes,
            "model_architecture" => "CNN_FaceRecognition_v1",
            "learning_rate" => CNNCheckinCore.LEARNING_RATE,
            "batch_size" => CNNCheckinCore.BATCH_SIZE,
            "visualization_paths" => viz_paths  # Incluir caminhos das visualizações
        )
        
        println("\n🎉 Pré-treino concluído!")
        println("📊 Resultados:")
        println("   - Melhor acurácia: $(round(best_val_acc*100, digits=2))% (Época $best_epoch)")
        println("   - Epochs treinadas: $(training_info["epochs_trained"])/$(CNNCheckinCore.PRETRAIN_EPOCHS)")
        println("   - Duração: $(round(duration_minutes, digits=1)) minutos")
        println("   - Visualizações geradas: $(length(viz_paths)) conjuntos")
        
        # Mostrar caminhos das visualizações
        if !isempty(viz_paths)
            println("\n🎨 Visualizações salvas em:")
            for viz_path in viz_paths
                println("   - $(basename(viz_path))")
            end
        end

        # Save model and configuration with TOML support
        success = save_pretrained_model_with_toml(model, person_names, CNNCheckinCore.MODEL_PATH, 
                                                 CNNCheckinCore.CONFIG_PATH, training_info)

        if success
            println("\n✅ Sistema pré-treinado salvo com sucesso!")
            println("\n📁 Arquivos gerados:")
            println("   - Configuração: $(CNNCheckinCore.CONFIG_PATH)")
            println("   - Modelo JLD2: $(CNNCheckinCore.MODEL_PATH)")
            println("   - Dados TOML: $(CNNCheckinCore.MODEL_DATA_TOML_PATH)")
            println("   - Pesos TOML: $WEIGHTS_TOML_PATH")
            
            # Criar relatório de visualizações
            if !isempty(viz_paths)
                create_training_visualization_report(viz_paths, person_names, training_info)
            end
            
            # Mostrar estatísticas dos arquivos de peso
            println("\n📈 Resumo dos Treinamentos:")
            list_saved_trainings(WEIGHTS_TOML_PATH)
        else
            println("❌ Modelo treinado mas alguns arquivos falharam ao salvar")
        end

        return success

    catch e
        println("❌ Erro durante pré-treino: $e")
        return false
    end
end

# Função para criar relatório das visualizações
function create_training_visualization_report(viz_paths::Vector{String}, 
                                            person_names::Vector{String}, 
                                            training_info::Dict)
    println("📋 Criando relatório das visualizações...")
    
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    report_path = joinpath(VIZ_OUTPUT_PATH, "training_report_$timestamp.md")
    
    report_content = """
# Relatório de Treinamento com Visualizações

**Data do Treinamento:** $(Dates.format(now(), "dd/mm/yyyy HH:MM:SS"))
**Duração:** $(round(training_info["duration_minutes"], digits=1)) minutos
**Melhor Acurácia:** $(round(training_info["final_accuracy"]*100, digits=2))% (Época $(training_info["best_epoch"]))

## Configurações do Treinamento
- **Arquitetura:** $(training_info["model_architecture"])
- **Épocas Treinadas:** $(training_info["epochs_trained"])
- **Learning Rate:** $(training_info["learning_rate"])
- **Batch Size:** $(training_info["batch_size"])
- **Augmentação:** $(training_info["augmentation_used"] ? "Sim" : "Não")

## Dataset
- **Pessoas:** $(length(person_names))
- **Imagens de Treino:** $(training_info["total_training_images"])
- **Imagens de Validação:** $(training_info["total_validation_images"])

### Pessoas Incluídas:
$(join(["- $name" for name in person_names], "\n"))

## Visualizações Geradas
$(join(["- $(basename(path))" for path in viz_paths], "\n"))

## Como Interpretar as Visualizações

### Camadas Convolucionais (Conv):
1. **Camada 1-2:** Detectam bordas, linhas e texturas básicas
2. **Camada 3-4:** Combinam características básicas em padrões mais complexos
3. **Camadas Finais:** Detectam características específicas de faces

### Filtros vs Ativações:
- **Filtros:** Mostram quais padrões cada filtro "procura"
- **Ativações:** Mostram como a imagem é "vista" após passar pela camada

### Indicadores de Qualidade:
- ✅ **Bom:** Filtros diversos, ativações claras e diferenciadas
- ⚠️ **Atenção:** Filtros muito similares ou ativações uniformes
- ❌ **Problema:** Filtros "mortos" (todos zeros) ou ativações caóticas

## Análise por Camada

### Layer 1 (Conv 3x3, 3→64):
- Deve mostrar detectores de bordas em diferentes orientações
- Ativações devem realçar contornos faciais

### Layer 2 (Conv 3x3, 64→128):  
- Filtros devem combinar bordas em padrões mais complexos
- Ativações começam a mostrar características faciais básicas

### Layer 3 (Conv 3x3, 128→256):
- Detectores de características faciais mais específicas
- Ativações mostram regiões importantes para reconhecimento

### Layer 4 (Conv 3x3, 256→256):
- Características muito específicas para cada pessoa
- Ativações altamente seletivas

### Layers Dense:
- Representação vetorial das características
- Visualizada como mapas de calor da "importância"

## Próximos Passos
1. Compare visualizações entre diferentes épocas
2. Analise se os filtros estão aprendendo características relevantes
3. Identifique possíveis overfitting observando ativações muito específicas
4. Use as visualizações para ajustar hiperparâmetros se necessário

## Arquivos de Referência
- **Modelo JLD2:** $(CNNCheckinCore.MODEL_PATH)
- **Configuração:** $(CNNCheckinCore.CONFIG_PATH)
- **Pesos TOML:** $WEIGHTS_TOML_PATH

---
*Relatório gerado automaticamente pelo sistema CNN Checkin*
"""
    
    try
        open(report_path, "w") do f
            write(f, report_content)
        end
        println("✅ Relatório salvo em: $report_path")
        return report_path
    catch e
        println("❌ Erro ao criar relatório: $e")
        return nothing
    end
end

# Save model with TOML support (mesmo que antes)
function save_pretrained_model_with_toml(model, person_names, model_path, config_path, training_info)
    # Salvar modelo em JLD2
    JLD2.save(model_path, "model", model, "person_names", person_names)
    
    # Salvar config com prefixo do módulo correto
    config = CNNCheckinCore.load_config(config_path)
    config["training"]["epochs_trained"] = training_info["epochs_trained"]
    config["training"]["final_accuracy"] = training_info["final_accuracy"]
    config["training"]["best_epoch"] = training_info["best_epoch"]
    config["data"]["person_names"] = person_names
    CNNCheckinCore.save_config(config, config_path)
    
    # Salvar dados TOML
    CNNCheckinCore.save_model_data_toml(model, person_names, CNNCheckinCore.MODEL_DATA_TOML_PATH)
    
    # Salvar pesos TOML
    save_pretrained_weights_toml(model, person_names, training_info, WEIGHTS_TOML_PATH)
    
    return true
end

# Menu para gerenciar visualizações
function visualization_management_menu()
    while true
        println("\n🎨 MENU DE GERENCIAMENTO DE VISUALIZAÇÕES")
        println("=" * 50)
        println("1 - Executar treinamento com visualizações")
        println("2 - Gerar visualizações de modelo existente")
        println("3 - Comparar visualizações entre treinamentos")
        println("4 - Listar visualizações disponíveis")
        println("5 - Limpar visualizações antigas")
        println("6 - Voltar")
        
        print("Escolha uma opção: ")
        choice = strip(readline())
        
        if choice == "1"
            print("Salvar visualizações a cada quantas épocas? (padrão: 5): ")
            freq_input = readline()
            save_freq = isempty(freq_input) ? 5 : parse(Int, freq_input)
            return pretrain_command_with_visualizations(save_freq)
            
        elseif choice == "2"
            println("🔍 Função para gerar visualizações de modelo existente")
            println("   (Requer implementação adicional para carregar modelo JLD2)")
            
        elseif choice == "3"
            println("📊 Comparar visualizações entre treinamentos")
            viz_dirs = []
            if isdir(VIZ_OUTPUT_PATH)
                viz_dirs = filter(isdir, [joinpath(VIZ_OUTPUT_PATH, d) for d in readdir(VIZ_OUTPUT_PATH)])
            end
            
            if length(viz_dirs) < 2
                println("❌ São necessárias pelo menos 2 visualizações para comparação")
            else
                println("Diretórios disponíveis:")
                for (i, dir) in enumerate(viz_dirs)
                    println("   $i - $(basename(dir))")
                end
                
                print("Escolha o primeiro diretório (número): ")
                idx1 = parse(Int, readline())
                print("Escolha o segundo diretório (número): ")
                idx2 = parse(Int, readline())
                
                if 1 <= idx1 <= length(viz_dirs) && 1 <= idx2 <= length(viz_dirs)
                    compare_layer_visualizations(viz_dirs[idx1], viz_dirs[idx2])
                end
            end
            
        elseif choice == "4"
            println("📋 Visualizações Disponíveis:")
            if isdir(VIZ_OUTPUT_PATH)
                for dir in readdir(VIZ_OUTPUT_PATH)
                    dir_path = joinpath(VIZ_OUTPUT_PATH, dir)
                    if isdir(dir_path)
                        println("   📁 $dir")
                        # Mostrar pessoas dentro do diretório
                        subdirs = filter(isdir, [joinpath(dir_path, d) for d in readdir(dir_path)])
                        if !isempty(subdirs)
                            println("      👥 Pessoas: $(join([basename(d) for d in subdirs[1:min(3, end)]], ", "))")
                        end
                    end
                end
            else
                println("   📭 Nenhuma visualização encontrada")
            end
            
        elseif choice == "5"
            println("🧹 Limpeza de Visualizações Antigas")
            if isdir(VIZ_OUTPUT_PATH)
                viz_dirs = filter(isdir, [joinpath(VIZ_OUTPUT_PATH, d) for d in readdir(VIZ_OUTPUT_PATH)])
                if length(viz_dirs) > 3
                    print("Manter quantas visualizações mais recentes? (padrão: 3): ")
                    keep_n = readline()
                    keep_n = isempty(keep_n) ? 3 : parse(Int, keep_n)
                    
                    # Ordenar por data de modificação
                    sort!(viz_dirs, by=d -> stat(d).mtime, rev=true)
                    
                    # Remover antigas
                    for old_dir in viz_dirs[keep_n+1:end]
                        try
                            rm(old_dir, recursive=true)
                            println("🗑️ Removido: $(basename(old_dir))")
                        catch e
                            println("❌ Erro ao remover $(basename(old_dir)): $e")
                        end
                    end
                    
                    println("✅ Limpeza concluída. Mantidas $(min(keep_n, length(viz_dirs))) visualizações.")
                else
                    println("ℹ️ Poucas visualizações para limpeza ($(length(viz_dirs)) encontradas)")
                end
            else
                println("📭 Diretório de visualizações não existe")
            end
            
        elseif choice == "6"
            return false
            
        else
            println("❌ Opção inválida!")
        end
        
        println("\nPressione Enter para continuar...")
        readline()
    end
end

# Export functions
export pretrain_command_with_visualizations, visualization_management_menu,
       create_training_visualization_report