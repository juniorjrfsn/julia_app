# file : visualizar_cnn_camadas.jl
# Descrição: Carrega imagens, treina uma CNN simples e visualiza as ativações das camadas convolucionais.
# Descrição: CNN aprimorada com visualizações avançadas, data augmentation,
# arquitetura otimizada e análises detalhadas
 


 
using Flux
using Images, ImageIO, FileIO
using Statistics
using ProgressMeter
using Random
using Plots
using LinearAlgebra
using StatsBase
using Colors
using Dates
using JSON3

# Tentar importar CUDA, mas continuar sem GPU se não disponível
try
    using CUDA
    global use_cuda = CUDA.functional()
    println("CUDA disponível: usando GPU")
catch
    global use_cuda = false
    println("CUDA não disponível, usando apenas CPU")
end

# === CONFIGURAÇÕES ===
const INPUT_DIR = "../../../dados/fotos"
const OUTPUT_DIR = "../../../dados/cnn/resultados_cnn_corrigido"
const IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
const TARGET_SIZE = (128, 128)
const BATCH_SIZE = 8
const EPOCHS = 30
const LEARNING_RATE = 0.001
const DROPOUT_RATE = 0.3

mkpath(OUTPUT_DIR)
Random.seed!(42)

# === ESTRUTURAS ===
struct TrainingMetrics
    train_losses::Vector{Float32}
    val_losses::Vector{Float32}
    val_accuracies::Vector{Float32}
    train_accuracies::Vector{Float32}
    learning_rates::Vector{Float32}
end

# === FUNÇÕES DE PRÉ-PROCESSAMENTO ===

"""
Aplica data augmentation básico nas imagens
"""
function simple_augmentation(img; flip=true, brightness=true)
    augmented = copy(img)
    
    # Flip horizontal
    if flip && rand() > 0.5
        augmented = reverse(augmented, dims=2)
    end
    
    # Ajuste de brilho
    if brightness && rand() > 0.5
        brightness_factor = 1.0 + (rand() - 0.5) * 0.3
        augmented = clamp.(augmented .* brightness_factor, 0f0, 1f0)
    end
    
    # Ruído leve
    if rand() > 0.7
        noise = randn(Float32, size(augmented)) .* 0.01f0
        augmented = clamp.(augmented .+ noise, 0f0, 1f0)
    end
    
    return augmented
end

"""
Carrega e pré-processa imagens com tratamento de erros robusto
"""
function load_and_preprocess_image(filepath; target_size=(128, 128), augment=false)
    try
        img = load(filepath)
        
        # Redimensionar
        img = imresize(img, target_size...)
        
        # Converter para array Float32
        if eltype(img) <: Gray
            img_array = Float32.(img)
            img_array = reshape(img_array, target_size..., 1)
            # Converter grayscale para RGB
            img_array = repeat(img_array, 1, 1, 3)
        elseif eltype(img) <: RGB
            img_array = Float32.(channelview(img))
            img_array = permutedims(img_array, (2, 3, 1))
        else
            @warn "Formato não suportado: $filepath"
            return nothing
        end
        
        # Normalização simples [0, 1]
        img_array = clamp.(img_array, 0f0, 1f0)
        
        # Data augmentation se solicitado
        if augment
            img_array = simple_augmentation(img_array)
        end
        
        return img_array
        
    catch e
        @warn "Erro ao carregar $filepath: $e"
        return nothing
    end
end

"""
Cria divisão balanceada dos dados
"""
function create_balanced_split(X, Y, train_ratio=0.8)
    n_samples = size(X, 4)
    indices = collect(1:n_samples)
    shuffle!(indices)
    
    n_train = Int(floor(n_samples * train_ratio))
    
    train_indices = indices[1:n_train]
    val_indices = indices[(n_train+1):end]
    
    X_train = X[:, :, :, train_indices]
    Y_train = Y[:, train_indices]
    X_val = X[:, :, :, val_indices]
    Y_val = Y[:, val_indices]
    
    return X_train, Y_train, X_val, Y_val
end

# === CARREGAMENTO DE DADOS ===

if !isdir(INPUT_DIR)
    error("Diretório não encontrado: $INPUT_DIR")
end

image_files = filter(f -> any(ext -> endswith(lowercase(f), ext), IMAGE_EXTENSIONS), readdir(INPUT_DIR))
println("Encontradas $(length(image_files)) imagens.")

if length(image_files) == 0
    error("Nenhuma imagem encontrada em: $INPUT_DIR")
end

# Carregamento das imagens
images = []
labels = []
valid_files = []

@showprogress "Carregando imagens..." for file in image_files
    path = joinpath(INPUT_DIR, file)
    img = load_and_preprocess_image(path; target_size=TARGET_SIZE)
    if img !== nothing
        push!(images, img)
        # Classificação baseada no nome do arquivo
        push!(labels, occursin("junior", lowercase(file)) ? 1 : 2)
        push!(valid_files, file)
    end
end

if length(images) == 0
    error("Nenhuma imagem válida foi carregada.")
end

println("$(length(images)) imagens carregadas com sucesso.")

# Converter para array 4D
X = cat(images..., dims=4)
println("Shape do dataset: ", size(X))

# Criar dados aumentados
println("Criando dados aumentados...")
X_augmented = similar(X, size(X, 1), size(X, 2), size(X, 3), size(X, 4) * 2)
labels_augmented = Vector{Int}(undef, length(labels) * 2)

# Dados originais
X_augmented[:, :, :, 1:size(X, 4)] = X
labels_augmented[1:length(labels)] = labels

# Dados aumentados
for i in 1:size(X, 4)
    augmented_img = simple_augmentation(X[:, :, :, i])
    X_augmented[:, :, :, size(X, 4) + i] = augmented_img
    labels_augmented[length(labels) + i] = labels[i]
end

X = X_augmented
labels = labels_augmented

println("Dataset final: $(size(X))")

# Criar rótulos one-hot
Y = Flux.onehotbatch(labels, 1:2)
label_distribution = [count(==(i), labels) for i in 1:2]
println("Distribuição: Junior=$(label_distribution[1]), Outros=$(label_distribution[2])")

# Dividir dados
X_train, Y_train, X_val, Y_val = create_balanced_split(X, Y, 0.8)
println("Treino: $(size(X_train, 4)) amostras")
println("Validação: $(size(X_val, 4)) amostras")

# === MODELO ===

function create_cnn_model()
    return Chain(
        # Bloco 1
        Conv((5, 5), 3 => 32, relu; pad=2),
        BatchNorm(32),
        MaxPool((2, 2)),
        Dropout(0.1),
        
        # Bloco 2  
        Conv((3, 3), 32 => 64, relu; pad=1),
        BatchNorm(64),
        MaxPool((2, 2)),
        Dropout(0.2),
        
        # Bloco 3
        Conv((3, 3), 64 => 128, relu; pad=1),
        BatchNorm(128),
        MaxPool((2, 2)),
        Dropout(0.3),
        
        # Bloco 4
        Conv((3, 3), 128 => 256, relu; pad=1),
        BatchNorm(256),
        AdaptiveMaxPool((4, 4)),
        Dropout(0.3),
        
        # Classificador
        Flux.flatten,
        Dense(256 * 4 * 4, 512, relu),
        Dropout(DROPOUT_RATE),
        Dense(512, 128, relu),
        Dropout(DROPOUT_RATE * 0.5),
        Dense(128, 2)
    )
end

model = create_cnn_model()
println("Modelo criado:")
println(model)

# Contar parâmetros
total_params = sum(length, Flux.trainable(model))
println("Total de parâmetros: $(total_params)")

# === FUNÇÕES DE TREINAMENTO ===

function accuracy(ŷ, y)
    return mean(Flux.onecold(ŷ, 1:2) .== Flux.onecold(y, 1:2))
end

function train_model!(model, X_train, Y_train, X_val, Y_val; epochs=EPOCHS, lr=LEARNING_RATE)
    # Otimizador
    opt_state = Flux.setup(Adam(lr), model)
    
    # Métricas
    metrics = TrainingMetrics(
        Float32[], Float32[], Float32[], Float32[], Float32[]
    )
    
    # Early stopping
    best_val_acc = 0.0
    best_model_state = deepcopy(Flux.state(model))
    patience = 0
    max_patience = 8
    
    println("Iniciando treinamento...")
    println("Épocas: $epochs, Learning Rate: $lr")
    
    # Progress bar
    prog = Progress(epochs, desc="Treinamento: ")
    
    for epoch in 1:epochs
        # === TREINAMENTO ===
        # Forward pass e cálculo do gradiente
        loss_val, grads = Flux.withgradient(model) do m
            ŷ = m(X_train)
            Flux.logitcrossentropy(ŷ, Y_train)
        end
        
        # Update dos parâmetros
        Flux.update!(opt_state, model, grads[1])
        
        train_loss = loss_val
        
        # Acurácia de treino
        ŷ_train = model(X_train)
        train_acc = accuracy(ŷ_train, Y_train)
        
        # === VALIDAÇÃO ===
        ŷ_val = model(X_val)
        val_loss = Flux.logitcrossentropy(ŷ_val, Y_val)
        val_acc = accuracy(ŷ_val, Y_val)
        
        # Salvar métricas
        push!(metrics.train_losses, train_loss)
        push!(metrics.val_losses, val_loss)
        push!(metrics.train_accuracies, train_acc)
        push!(metrics.val_accuracies, val_acc)
        push!(metrics.learning_rates, lr)
        
        # Early stopping
        if val_acc > best_val_acc
            best_val_acc = val_acc
            best_model_state = deepcopy(Flux.state(model))
            patience = 0
        else
            patience += 1
        end
        
        # Log a cada 3 épocas
        if epoch % 3 == 0 || epoch == 1 || epoch == epochs
            println("\nÉpoca $epoch:")
            println("  Train: Loss=$(round(train_loss, digits=4)), Acc=$(round(train_acc*100, digits=2))%")
            println("  Val:   Loss=$(round(val_loss, digits=4)), Acc=$(round(val_acc*100, digits=2))%")
            println("  Melhor Val Acc: $(round(best_val_acc*100, digits=2))%")
        end
        
        # Update progress
        ProgressMeter.update!(prog, epoch)
        
        # Early stopping
        if patience >= max_patience
            println("\nEarly stopping na época $epoch")
            println("Melhor acurácia: $(round(best_val_acc*100, digits=2))%")
            break
        end
    end
    
    # Restaurar melhor modelo
    println("\nRestaurando melhor modelo...")
    Flux.loadmodel!(model, best_model_state)
    
    return metrics
end

# === FUNÇÕES DE SERIALIZAÇÃO DE PARÂMETROS ===

"""
Converte arrays para formato serializável (listas aninhadas)
"""
function array_to_serializable(arr::AbstractArray)
    if ndims(arr) == 1
        return collect(arr)
    elseif ndims(arr) == 2
        return [collect(arr[i, :]) for i in 1:size(arr, 1)]
    elseif ndims(arr) == 3
        return [[[arr[i, j, k] for k in 1:size(arr, 3)] for j in 1:size(arr, 2)] for i in 1:size(arr, 1)]
    elseif ndims(arr) == 4
        return [[[[arr[i, j, k, l] for l in 1:size(arr, 4)] for k in 1:size(arr, 3)] for j in 1:size(arr, 2)] for i in 1:size(arr, 1)]
    else
        return collect(arr)  # Fallback
    end
end

"""
Extrai parâmetros de uma camada específica
"""
function extract_layer_params(layer, layer_name, layer_index)
    params = Dict{String, Any}(
        "layer_name" => layer_name,
        "layer_index" => layer_index,
        "layer_type" => string(typeof(layer))
    )
    
    if hasfield(typeof(layer), :weight) && !isnothing(layer.weight)
        params["weights"] = Dict(
            "shape" => collect(size(layer.weight)),
            "data" => array_to_serializable(layer.weight)
        )
    end
    
    if hasfield(typeof(layer), :bias) && !isnothing(layer.bias)
        params["bias"] = Dict(
            "shape" => collect(size(layer.bias)),
            "data" => array_to_serializable(layer.bias)
        )
    end
    
    # Parâmetros específicos para BatchNorm
    if isa(layer, BatchNorm)
        if hasfield(typeof(layer), :μ) && !isnothing(layer.μ)
            params["running_mean"] = Dict(
                "shape" => collect(size(layer.μ)),
                "data" => array_to_serializable(layer.μ)
            )
        end
        if hasfield(typeof(layer), :σ²) && !isnothing(layer.σ²)
            params["running_var"] = Dict(
                "shape" => collect(size(layer.σ²)),
                "data" => array_to_serializable(layer.σ²)
            )
        end
        if hasfield(typeof(layer), :γ) && !isnothing(layer.γ)
            params["gamma"] = Dict(
                "shape" => collect(size(layer.γ)),
                "data" => array_to_serializable(layer.γ)
            )
        end
        if hasfield(typeof(layer), :β) && !isnothing(layer.β)
            params["beta"] = Dict(
                "shape" => collect(size(layer.β)),
                "data" => array_to_serializable(layer.β)
            )
        end
    end
    
    # Parâmetros específicos para Conv
    if isa(layer, Conv)
        params["kernel_size"] = layer.σ == relu ? "relu" : string(layer.σ)
        params["stride"] = hasfield(typeof(layer), :stride) ? layer.stride : (1, 1)
        params["padding"] = hasfield(typeof(layer), :pad) ? layer.pad : (0, 0)
    end
    
    return params
end

"""
Salva todos os parâmetros do modelo em formato JSON
"""
function save_model_parameters(model, output_dir, metrics=nothing)
    params_dir = joinpath(output_dir, "model_parameters")
    mkpath(params_dir)
    
    # Metadados do modelo
    model_metadata = Dict{String, Any}(
        "model_name" => "CNN_Binary_Classifier",
        "creation_date" => string(Dates.now()),
        "total_parameters" => sum(length, Flux.trainable(model)),
        "input_shape" => [128, 128, 3],
        "output_classes" => 2,
        "class_names" => ["Junior", "Outros"]
    )
    
    # Extrair parâmetros de cada camada
    layers_params = []
    conv_count = 0
    dense_count = 0
    bn_count = 0
    
    for (i, layer) in enumerate(model.layers)
        if isa(layer, Conv)
            conv_count += 1
            layer_name = "conv_$conv_count"
        elseif isa(layer, Dense)
            dense_count += 1
            layer_name = "dense_$dense_count"
        elseif isa(layer, BatchNorm)
            bn_count += 1
            layer_name = "batchnorm_$bn_count"
        elseif isa(layer, MaxPool)
            layer_name = "maxpool_$i"
        elseif isa(layer, AdaptiveMaxPool)
            layer_name = "adaptive_maxpool_$i"
        elseif isa(layer, Dropout)
            layer_name = "dropout_$i"
        else
            layer_name = "layer_$i"
        end
        
        layer_params = extract_layer_params(layer, layer_name, i)
        push!(layers_params, layer_params)
    end
    
    # Estrutura completa
    full_model_data = Dict{String, Any}(
        "metadata" => model_metadata,
        "architecture" => layers_params
    )
    
    # Adicionar métricas de treinamento se disponíveis
    if metrics !== nothing
        training_data = Dict{String, Any}(
            "training_metrics" => Dict(
                "train_losses" => collect(metrics.train_losses),
                "val_losses" => collect(metrics.val_losses),
                "train_accuracies" => collect(metrics.train_accuracies),
                "val_accuracies" => collect(metrics.val_accuracies),
                "learning_rates" => collect(metrics.learning_rates),
                "best_val_accuracy" => maximum(metrics.val_accuracies),
                "final_train_accuracy" => metrics.train_accuracies[end],
                "total_epochs" => length(metrics.train_losses)
            )
        )
        full_model_data["training"] = training_data
    end
    
    # Salvar arquivo principal
    main_file = joinpath(params_dir, "model_complete.json")
    try
        open(main_file, "w") do f
            JSON3.pretty(f, full_model_data)
        end
        println("Parâmetros completos salvos: $main_file")
    catch e
        println("Erro ao salvar JSON principal: $e")
    end
    
    # Salvar arquivos separados por tipo de camada
    try
        # Apenas camadas convolucionais
        conv_layers = filter(p -> occursin("conv", p["layer_name"]), layers_params)
        if !isempty(conv_layers)
            conv_file = joinpath(params_dir, "conv_layers.json")
            open(conv_file, "w") do f
                JSON3.pretty(f, Dict("conv_layers" => conv_layers))
            end
        end
        
        # Apenas camadas densas
        dense_layers = filter(p -> occursin("dense", p["layer_name"]), layers_params)
        if !isempty(dense_layers)
            dense_file = joinpath(params_dir, "dense_layers.json")
            open(dense_file, "w") do f
                JSON3.pretty(f, Dict("dense_layers" => dense_layers))
            end
        end
        
        # Apenas BatchNorm
        bn_layers = filter(p -> occursin("batchnorm", p["layer_name"]), layers_params)
        if !isempty(bn_layers)
            bn_file = joinpath(params_dir, "batchnorm_layers.json")
            open(bn_file, "w") do f
                JSON3.pretty(f, Dict("batchnorm_layers" => bn_layers))
            end
        end
        
        println("Arquivos separados por tipo de camada salvos")
        
    catch e
        println("Erro ao salvar arquivos separados: $e")
    end
    
    # Salvar sumário compacto
    try
        summary_data = Dict{String, Any}(
            "model_summary" => Dict(
                "total_layers" => length(model.layers),
                "conv_layers" => conv_count,
                "dense_layers" => dense_count,
                "batchnorm_layers" => bn_count,
                "total_parameters" => model_metadata["total_parameters"],
                "trainable_params" => sum(length, Flux.trainable(model))
            )
        )
        
        if metrics !== nothing
            summary_data["performance"] = Dict(
                "best_val_acc" => round(maximum(metrics.val_accuracies) * 100, digits=2),
                "final_train_acc" => round(metrics.train_accuracies[end] * 100, digits=2),
                "epochs_trained" => length(metrics.train_losses)
            )
        end
        
        summary_file = joinpath(params_dir, "model_summary.json")
        open(summary_file, "w") do f
            JSON3.pretty(f, summary_data)
        end
        println("Sumário do modelo salvo: $summary_file")
        
    catch e
        println("Erro ao salvar sumário: $e")
    end
    
    return params_dir
end

# === TREINAMENTO ===
metrics = train_model!(model, X_train, Y_train, X_val, Y_val)

# === FUNÇÕES DE VISUALIZAÇÃO ===

"""
Extrai ativações das camadas convolucionais
"""
function extract_conv_activations(model, input)
    activations = []
    x = input
    
    for (i, layer) in enumerate(model.layers)
        x = layer(x)
        
        # Salvar ativações de Conv layers
        if isa(layer, Conv)
            push!(activations, (
                layer_idx=i,
                layer_name="Conv_$i",
                activation=copy(x),
                layer=layer
            ))
        end
    end
    
    return activations
end

"""
Salva mapas de características
"""
function save_feature_maps(activations, output_dir; max_filters=16)
    mkpath(output_dir)
    
    for act in activations
        layer_name = act.layer_name
        feature_maps = act.activation
        layer_dir = joinpath(output_dir, "feature_maps_$layer_name")
        mkpath(layer_dir)
        
        h, w, num_filters, batch_size = size(feature_maps)
        
        println("Salvando $layer_name: $(h)×$(w)×$num_filters")
        
        # Para cada batch (máximo 3)
        for batch in 1:min(batch_size, 3)
            batch_maps = feature_maps[:, :, :, batch]
            
            # Selecionar filtros mais ativos
            filter_activities = [maximum(abs, batch_maps[:, :, f]) for f in 1:num_filters]
            top_filters = sortperm(filter_activities, rev=true)[1:min(max_filters, num_filters)]
            
            # Criar grade 4x4
            grid_size = 4
            combined_img = zeros(Float32, h * grid_size, w * grid_size)
            
            for (idx, filter_idx) in enumerate(top_filters)
                if idx > 16 break end
                
                row = div(idx - 1, grid_size) + 1
                col = mod(idx - 1, grid_size) + 1
                
                fmap = batch_maps[:, :, filter_idx]
                
                # Normalização
                fmap_min, fmap_max = extrema(fmap)
                if fmap_max > fmap_min
                    fmap_norm = (fmap .- fmap_min) ./ (fmap_max - fmap_min)
                else
                    fmap_norm = zeros(Float32, size(fmap))
                end
                
                # Posição na grade
                r_start = (row - 1) * h + 1
                r_end = row * h
                c_start = (col - 1) * w + 1
                c_end = col * w
                
                combined_img[r_start:r_end, c_start:c_end] = fmap_norm
            end
            
            # Salvar
            grid_img = Gray.(combined_img)
            filename = "$(layer_name)_batch_$(batch).png"
            save(joinpath(layer_dir, filename), grid_img)
        end
    end
end

"""
Salva filtros das camadas convolucionais
"""
function save_conv_filters(model, output_dir)
    filters_dir = joinpath(output_dir, "conv_filters")
    mkpath(filters_dir)
    
    conv_idx = 1
    for (i, layer) in enumerate(model.layers)
        if isa(layer, Conv)
            filters = layer.weight
            h, w, in_ch, out_ch = size(filters)
            
            layer_name = "Conv_$(conv_idx)_layer_$i"
            layer_dir = joinpath(filters_dir, layer_name)
            mkpath(layer_dir)
            
            println("Salvando filtros $layer_name: $(h)×$(w)×$(in_ch)→$(out_ch)")
            
            # Salvar os 16 primeiros filtros
            for filter_idx in 1:min(16, out_ch)
                filter_data = filters[:, :, :, filter_idx]
                
                if in_ch == 3
                    # Filtro RGB
                    filter_norm = similar(filter_data)
                    for ch in 1:3
                        channel = filter_data[:, :, ch]
                        ch_min, ch_max = extrema(channel)
                        if ch_max > ch_min
                            filter_norm[:, :, ch] = (channel .- ch_min) ./ (ch_max - ch_min)
                        else
                            filter_norm[:, :, ch] = zeros(Float32, size(channel))
                        end
                    end
                    img = RGB.(filter_norm[:,:,1], filter_norm[:,:,2], filter_norm[:,:,3])
                else
                    # Média dos canais para visualização
                    filter_avg = mean(filter_data, dims=3)[:, :, 1]
                    filt_min, filt_max = extrema(filter_avg)
                    if filt_max > filt_min
                        filter_norm = (filter_avg .- filt_min) ./ (filt_max - filt_min)
                    else
                        filter_norm = zeros(Float32, size(filter_avg))
                    end
                    img = Gray.(filter_norm)
                end
                
                filename = "filter_$(filter_idx).png"
                save(joinpath(layer_dir, filename), img)
            end
            
            conv_idx += 1
        end
    end
end

"""
Cria gráficos de métricas
"""
function create_training_plots(metrics, output_dir)
    try
        # Plot 1: Loss e Acurácia
        p1 = plot(layout=(2, 1), size=(800, 600))
        
        # Loss
        plot!(p1[1], metrics.train_losses, label="Train Loss", linewidth=2, color=:blue)
        plot!(p1[1], metrics.val_losses, label="Val Loss", linewidth=2, color=:red)
        xlabel!(p1[1], "Época")
        ylabel!(p1[1], "Loss")
        title!(p1[1], "Evolução do Loss")
        
        # Acurácia
        plot!(p1[2], metrics.train_accuracies .* 100, label="Train Acc", linewidth=2, color=:green)
        plot!(p1[2], metrics.val_accuracies .* 100, label="Val Acc", linewidth=2, color=:orange)
        xlabel!(p1[2], "Época")
        ylabel!(p1[2], "Acurácia (%)")
        title!(p1[2], "Evolução da Acurácia")
        
        savefig(p1, joinpath(output_dir, "training_metrics.png"))
        
        println("Gráficos salvos em $output_dir")
        
    catch e
        println("Erro ao criar gráficos: $e")
    end
end

"""
Avaliação detalhada do modelo
"""
function evaluate_model(model, X, Y, output_dir)
    # Predições
    predictions = model(X)
    predicted_labels = Flux.onecold(predictions, 1:2)
    actual_labels = Flux.onecold(Y, 1:2)
    
    # Acurácia
    acc = mean(predicted_labels .== actual_labels)
    
    # Matriz de confusão
    conf_matrix = zeros(Int, 2, 2)
    for i in 1:length(predicted_labels)
        conf_matrix[actual_labels[i], predicted_labels[i]] += 1
    end
    
    # Métricas por classe
    precision = [conf_matrix[i, i] / max(sum(conf_matrix[:, i]), 1) for i in 1:2]
    recall = [conf_matrix[i, i] / max(sum(conf_matrix[i, :]), 1) for i in 1:2]
    f1_score = 2 .* (precision .* recall) ./ (precision .+ recall .+ 1e-8)
    
    # Salvar relatório
    report_path = joinpath(output_dir, "evaluation_report.txt")
    open(report_path, "w") do f
        println(f, "RELATÓRIO DE AVALIAÇÃO")
        println(f, repeat("=", 40))
        println(f, "Data: $(Dates.now())")
        println(f, "")
        
        println(f, "MÉTRICAS GERAIS:")
        println(f, "Acurácia: $(round(acc*100, digits=2))%")
        println(f, "Total de amostras: $(length(actual_labels))")
        println(f, "")
        
        println(f, "MATRIZ DE CONFUSÃO:")
        println(f, "        Predito")
        println(f, "      Junior | Outros")
        println(f, "Junior  $(conf_matrix[1,1])    |    $(conf_matrix[1,2])")
        println(f, "Outros  $(conf_matrix[2,1])    |    $(conf_matrix[2,2])")
        println(f, "")
        
        println(f, "MÉTRICAS POR CLASSE:")
        classes = ["Junior", "Outros"]
        for (i, class) in enumerate(classes)
            println(f, "$class:")
            println(f, "  Precisão: $(round(precision[i]*100, digits=2))%")
            println(f, "  Recall: $(round(recall[i]*100, digits=2))%")
            println(f, "  F1-Score: $(round(f1_score[i]*100, digits=2))%")
        end
    end
    
    println("Relatório salvo: $report_path")
    return acc, conf_matrix, precision, recall, f1_score
end

# === EXECUÇÃO DAS ANÁLISES ===

println("\n" * repeat("=", 50))
println("INICIANDO ANÁLISE DE VISUALIZAÇÕES")
println(repeat("=", 50))

# Extrair ativações
println("\nExtraindo ativações...")
n_samples = min(4, size(X, 4))
sample_X = X[:, :, :, 1:n_samples]
activations = extract_conv_activations(model, sample_X)

println("Encontradas $(length(activations)) camadas convolucionais")

# Salvar visualizações
println("\nSalvando mapas de características...")
save_feature_maps(activations, OUTPUT_DIR)

println("\nSalvando filtros...")
save_conv_filters(model, OUTPUT_DIR)

# Salvar parâmetros do modelo em JSON
println("\nSalvando parâmetros do modelo em JSON...")
params_dir = save_model_parameters(model, OUTPUT_DIR, metrics)

# Gráficos de treinamento
println("\nCriando gráficos...")
create_training_plots(metrics, OUTPUT_DIR)

# Avaliação
println("\nAvaliação do modelo...")
accuracy_final, conf_matrix, precision, recall, f1_score = evaluate_model(model, X, Y, OUTPUT_DIR)

# Relatório final
println("\nCriando relatório final...")
final_report = joinpath(OUTPUT_DIR, "RELATORIO_FINAL.md")
open(final_report, "w") do f
    println(f, "# RELATÓRIO FINAL - CNN")
    println(f, "")
    println(f, "**Data:** $(Dates.now())")
    println(f, "")
    
    println(f, "## RESUMO")
    println(f, "- **Acurácia Final:** $(round(accuracy_final*100, digits=2))%")
    println(f, "- **Parâmetros:** $total_params")
    println(f, "- **Épocas:** $(length(metrics.train_losses))")
    println(f, "- **Dataset:** $(size(X, 4)) amostras")
    println(f, "")
    
    println(f, "## PERFORMANCE")
    println(f, "### Métricas por Classe")
    classes = ["Junior", "Outros"]
    for (i, class) in enumerate(classes)
        println(f, "- **$class:**")
        println(f, "  - Precisão: $(round(precision[i]*100, digits=2))%")
        println(f, "  - Recall: $(round(recall[i]*100, digits=2))%")
        println(f, "  - F1-Score: $(round(f1_score[i]*100, digits=2))%")
    end
    println(f, "")
    
    println(f, "### Matriz de Confusão")
    println(f, "```")
    println(f, "        Predito")
    println(f, "      Junior | Outros")
    println(f, "Junior  $(conf_matrix[1,1])    |    $(conf_matrix[1,2])")
    println(f, "Outros  $(conf_matrix[2,1])    |    $(conf_matrix[2,2])")
    println(f, "```")
    println(f, "")
    
    println(f, "## ARQUIVOS GERADOS")
    println(f, "- `training_metrics.png` - Gráficos de treinamento")
    println(f, "- `feature_maps_Conv_*/` - Mapas de características")
    println(f, "- `conv_filters/` - Filtros das camadas")
    println(f, "- `evaluation_report.txt` - Relatório detalhado")
    println(f, "- `model_parameters/` - **NOVO:** Parâmetros do modelo em JSON")
    println(f, "  - `model_complete.json` - Modelo completo com todos os parâmetros")
    println(f, "  - `model_summary.json` - Sumário compacto do modelo")
    println(f, "  - `conv_layers.json` - Apenas camadas convolucionais")
    println(f, "  - `dense_layers.json` - Apenas camadas densas")
    println(f, "  - `batchnorm_layers.json` - Apenas camadas BatchNorm")
end

# === RESULTADOS FINAIS ===
println("\n" * repeat("=", 50))
println("ANÁLISE COMPLETA!")
println(repeat("=", 50))

println("\nRESULTADOS:")
println("  Acurácia Final: $(round(accuracy_final*100, digits=2))%")
println("  Melhor Val Acc: $(round(maximum(metrics.val_accuracies)*100, digits=2))%")
println("  Épocas: $(length(metrics.train_losses))")
println("  Parâmetros: $total_params")
println("  Amostras: $(size(X, 4))")

println("\nARQUIVOS SALVOS EM: $OUTPUT_DIR")
println("  ├── training_metrics.png")
println("  ├── evaluation_report.txt") 
println("  ├── RELATORIO_FINAL.md")
println("  ├── feature_maps_Conv_*/")
println("  ├── conv_filters/")
println("  └── model_parameters/")
println("      ├── model_complete.json")
println("      ├── model_summary.json")
println("      ├── conv_layers.json")
println("      ├── dense_layers.json")
println("      └── batchnorm_layers.json")

println("\n📊 PARÂMETROS SALVOS EM JSON:")
println("  ✓ Pesos de todas as camadas")
println("  ✓ Biases de todas as camadas") 
println("  ✓ Parâmetros BatchNorm (γ, β, μ, σ²)")
println("  ✓ Métricas de treinamento")
println("  ✓ Metadados do modelo")
println("  ✓ Arquivos separados por tipo de camada")

if accuracy_final > 0.8
    println("\nEXCELENTE! Modelo com alta performance!")
elseif accuracy_final > 0.7
    println("\nBOA performance do modelo!")
else
    println("\nModelo treinado.")
end

println("\n" * repeat("=", 50))





# # Para executar o script, use o comando:
# ## julia visualizar_cnn_camadas.jl
 
# using JSON3

# # Carregar modelo completo
# model_data = JSON3.read("model_complete.json")

# # Carregar apenas sumário
# summary = JSON3.read("model_summary.json")

# # Carregar apenas camadas específicas
# conv_layers = JSON3.read("conv_layers.json")
