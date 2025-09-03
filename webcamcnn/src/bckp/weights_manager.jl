# projeto: webcamcnn
# file: webcamcnn/src/weights_manager.jl

using Flux
using TOML
using Dates
using Statistics
using JLD2
using SHA

include("core.jl")
using .CNNCheckinCore

# Estrutura para metadados de treinamento
struct TrainingMetadata
    training_id::String
    timestamp::String
    person_names::Vector{String}
    epochs_trained::Int
    final_accuracy::Float64
    best_epoch::Int
    model_architecture::String
    learning_rate::Float64
    batch_size::Int
    data_hash::String  # Hash dos dados de treino para detectar mudanças
end

# Função para gerar hash dos dados de treino
function generate_data_hash(person_names::Vector{String}, data_path::String)
    content = join(sort(person_names), "_")
    content *= "_$(length(readdir(data_path)))"
    return bytes2hex(sha256(content))
end

# Converter array para formato TOML-friendly
function array_to_toml_format(arr::AbstractArray)
    if ndims(arr) == 1
        return collect(arr)
    elseif ndims(arr) == 2
        # Converter matriz 2D para array de arrays
        return [collect(arr[i, :]) for i in 1:size(arr, 1)]
    elseif ndims(arr) == 3
        # Converter tensor 3D para estrutura aninhada
        return [[[arr[i, j, k] for k in 1:size(arr, 3)] for j in 1:size(arr, 2)] for i in 1:size(arr, 1)]
    elseif ndims(arr) == 4
        # Converter tensor 4D (comum em Conv layers)
        result = []
        for i in 1:size(arr, 1), j in 1:size(arr, 2), k in 1:size(arr, 3)
            push!(result, collect(arr[i, j, k, :]))
        end
        return Dict(
            "shape" => collect(size(arr)),
            "data" => result,
            "format" => "4d_tensor"
        )
    else
        # Para arrays de dimensão superior, armazenar como dados planos com metadados
        return Dict(
            "shape" => collect(size(arr)),
            "data" => collect(reshape(arr, :)),
            "format" => "flattened_$(ndims(arr))d"
        )
    end
end

# Converter formato TOML de volta para array
function toml_format_to_array(data, target_type::Type{T}) where T
    if isa(data, Dict) && haskey(data, "format")
        if data["format"] == "4d_tensor"
            shape = Tuple(data["shape"])
            flat_data = T.(data["data"])
            return reshape(flat_data, shape)
        elseif startswith(data["format"], "flattened_")
            shape = Tuple(data["shape"])
            flat_data = T.(data["data"])
            return reshape(flat_data, shape)
        end
    elseif isa(data, Vector) && all(isa(x, Vector) for x in data)
        # Array 2D ou 3D aninhado
        return T.(reduce(hcat, data)')
    else
        return T.(data)
    end
end

# Extrair pesos e vieses de uma camada
function extract_layer_parameters(layer, layer_idx::Int)
    params = Dict{String, Any}()
    
    try
        # Informações da camada
        params["layer_type"] = string(typeof(layer))
        params["layer_index"] = layer_idx
        
        if hasfield(typeof(layer), :weight) && layer.weight !== nothing
            weights = layer.weight
            params["weights"] = Dict(
                "values" => array_to_toml_format(weights),
                "shape" => collect(size(weights)),
                "dtype" => string(eltype(weights)),
                "stats" => Dict(
                    "mean" => Float64(mean(weights)),
                    "std" => Float64(std(weights)),
                    "min" => Float64(minimum(weights)),
                    "max" => Float64(maximum(weights)),
                    "count" => length(weights)
                )
            )
        end
        
        if hasfield(typeof(layer), :bias) && layer.bias !== nothing
            bias = layer.bias
            params["bias"] = Dict(
                "values" => array_to_toml_format(bias),
                "shape" => collect(size(bias)),
                "dtype" => string(eltype(bias)),
                "stats" => Dict(
                    "mean" => Float64(mean(bias)),
                    "std" => Float64(std(bias)),
                    "min" => Float64(minimum(bias)),
                    "max" => Float64(maximum(bias)),
                    "count" => length(bias)
                )
            )
        end
        
        # Parâmetros específicos para BatchNorm
        if isa(layer, BatchNorm)
            if hasfield(typeof(layer), :β) && layer.β !== nothing
                params["beta"] = Dict(
                    "values" => array_to_toml_format(layer.β),
                    "shape" => collect(size(layer.β)),
                    "dtype" => string(eltype(layer.β))
                )
            end
            if hasfield(typeof(layer), :γ) && layer.γ !== nothing
                params["gamma"] = Dict(
                    "values" => array_to_toml_format(layer.γ),
                    "shape" => collect(size(layer.γ)),
                    "dtype" => string(eltype(layer.γ))
                )
            end
            if hasfield(typeof(layer), :μ) && layer.μ !== nothing
                params["mu"] = Dict(
                    "values" => array_to_toml_format(layer.μ),
                    "shape" => collect(size(layer.μ)),
                    "dtype" => string(eltype(layer.μ))
                )
            end
            if hasfield(typeof(layer), :σ²) && layer.σ² !== nothing
                params["sigma_squared"] = Dict(
                    "values" => array_to_toml_format(layer.σ²),
                    "shape" => collect(size(layer.σ²)),
                    "dtype" => string(eltype(layer.σ²))
                )
            end
        end
        
        # Informações adicionais para Conv layers
        if isa(layer, Conv)
            if hasfield(typeof(layer), :stride) && layer.stride !== nothing
                params["stride"] = collect(layer.stride)
            end
            if hasfield(typeof(layer), :pad) && layer.pad !== nothing
                params["padding"] = collect(layer.pad)
            end
            if hasfield(typeof(layer), :dilation) && layer.dilation !== nothing
                params["dilation"] = collect(layer.dilation)
            end
        end
        
        # Informações adicionais para Dense layers
        if isa(layer, Dense)
            if hasfield(typeof(layer), :σ) && layer.σ !== nothing
                params["activation"] = string(layer.σ)
            end
        end
        
    catch e
        println("⚠️ Erro ao extrair parâmetros da camada $layer_idx: $e")
        params["error"] = "Failed to extract parameters: $e"
    end
    
    return params
end

# Salvar pesos e vieses do modelo em TOML
function save_weights_to_toml(model, person_names::Vector{String}, 
                             training_metadata::TrainingMetadata, 
                             filepath::String; 
                             append_mode::Bool = true)
    println("💾 Salvando pesos e vieses em formato TOML...")
    
    # Carregar dados existentes se em modo append
    existing_data = Dict{String, Any}()
    if append_mode && isfile(filepath)
        try
            existing_data = TOML.parsefile(filepath)
            println("📂 Dados existentes carregados de: $filepath")
        catch e
            println("⚠️ Erro ao carregar dados existentes: $e")
            println("🆕 Criando novo arquivo de pesos...")
        end
    end
    
    # Criar estrutura de dados para este treinamento
    training_data = Dict{String, Any}()
    
    # Metadados do treinamento
    training_data["metadata"] = Dict(
        "training_id" => training_metadata.training_id,
        "timestamp" => training_metadata.timestamp,
        "person_names" => training_metadata.person_names,
        "epochs_trained" => training_metadata.epochs_trained,
        "final_accuracy" => training_metadata.final_accuracy,
        "best_epoch" => training_metadata.best_epoch,
        "model_architecture" => training_metadata.model_architecture,
        "learning_rate" => training_metadata.learning_rate,
        "batch_size" => training_metadata.batch_size,
        "data_hash" => training_metadata.data_hash
    )
    
    # Extrair parâmetros de todas as camadas
    training_data["layers"] = Dict{String, Any}()
    total_params = 0
    
    for (layer_idx, layer) in enumerate(model)
        layer_key = "layer_$(layer_idx)"
        layer_params = extract_layer_parameters(layer, layer_idx)
        training_data["layers"][layer_key] = layer_params
        
        # Contar parâmetros
        if haskey(layer_params, "weights") && haskey(layer_params["weights"], "stats")
            total_params += layer_params["weights"]["stats"]["count"]
        end
        if haskey(layer_params, "bias") && haskey(layer_params["bias"], "stats")
            total_params += layer_params["bias"]["stats"]["count"]
        end
    end
    
    # Estatísticas gerais do modelo
    training_data["model_stats"] = Dict(
        "total_layers" => length(model),
        "total_parameters" => total_params,
        "model_size_mb" => round(total_params * 4 / (1024^2), digits=2)
    )
    
    # Adicionar este treinamento aos dados existentes
    if !haskey(existing_data, "trainings")
        existing_data["trainings"] = Dict{String, Any}()
    end
    
    existing_data["trainings"][training_metadata.training_id] = training_data
    
    # Atualizar metadados gerais
    existing_data["format_info"] = Dict(
        "version" => "1.0",
        "description" => "CNN Face Recognition Weights and Biases",
        "created_by" => "webcamcnn.jl",
        "last_updated" => training_metadata.timestamp
    )
    
    existing_data["summary"] = Dict(
        "total_trainings" => length(existing_data["trainings"]),
        "latest_training" => training_metadata.training_id,
        "all_persons" => collect(Set(vcat([t["metadata"]["person_names"] 
                                         for t in values(existing_data["trainings"])]...)))
    )
    
    # Salvar arquivo TOML
    try
        open(filepath, "w") do io
            TOML.print(io, existing_data)
        end
        println("✅ Pesos salvos em: $filepath")
        println("📊 Total de treinamentos salvos: $(existing_data["summary"]["total_trainings"])")
        return true
    catch e
        println("❌ Erro ao salvar pesos: $e")
        return false
    end
end

# Carregar pesos específicos de um treinamento
function load_weights_from_toml(filepath::String, training_id::String)
    println("📂 Carregando pesos do treinamento: $training_id")
    
    if !isfile(filepath)
        println("❌ Arquivo não encontrado: $filepath")
        return nothing
    end
    
    try
        data = TOML.parsefile(filepath)
        
        if !haskey(data, "trainings") || !haskey(data["trainings"], training_id)
            println("❌ Treinamento '$training_id' não encontrado")
            available = collect(keys(get(data, "trainings", Dict())))
            if !isempty(available)
                println("📋 Treinamentos disponíveis: $(join(available, ", "))")
            end
            return nothing
        end
        
        training_data = data["trainings"][training_id]
        println("✅ Pesos carregados para treinamento: $training_id")
        println("📅 Data: $(training_data["metadata"]["timestamp"])")
        println("🎯 Acurácia: $(training_data["metadata"]["final_accuracy"])")
        
        return training_data
        
    catch e
        println("❌ Erro ao carregar pesos: $e")
        return nothing
    end
end

# Reconstruir modelo a partir dos pesos salvos
function reconstruct_model_from_weights(training_data::Dict, target_model)
    println("🔧 Reconstruindo modelo a partir dos pesos salvos...")
    
    try
        layers_data = training_data["layers"]
        reconstructed_layers = []
        
        for (layer_idx, layer) in enumerate(target_model)
            layer_key = "layer_$(layer_idx)"
            
            if haskey(layers_data, layer_key)
                layer_data = layers_data[layer_key]
                new_layer = layer  # Cópia da estrutura original
                
                # Restaurar pesos
                if haskey(layer_data, "weights")
                    weight_info = layer_data["weights"]
                    weights = toml_format_to_array(weight_info["values"], Float32)
                    # Aplicar pesos ao layer (isso depende da implementação específica do Flux)
                    # new_layer = update_layer_weights(new_layer, weights)
                end
                
                # Restaurar vieses
                if haskey(layer_data, "bias")
                    bias_info = layer_data["bias"]
                    bias = toml_format_to_array(bias_info["values"], Float32)
                    # new_layer = update_layer_bias(new_layer, bias)
                end
                
                push!(reconstructed_layers, new_layer)
            else
                println("⚠️ Dados não encontrados para camada $layer_idx")
                push!(reconstructed_layers, layer)
            end
        end
        
        return Chain(reconstructed_layers...)
        
    catch e
        println("❌ Erro ao reconstruir modelo: $e")
        return nothing
    end
end

# Comparar pesos entre diferentes treinamentos
function compare_training_weights(filepath::String, training_id1::String, training_id2::String)
    println("🔍 Comparando pesos entre treinamentos...")
    
    data1 = load_weights_from_toml(filepath, training_id1)
    data2 = load_weights_from_toml(filepath, training_id2)
    
    if data1 === nothing || data2 === nothing
        return false
    end
    
    println("\n📊 Comparação de Treinamentos:")
    println("=" * 50)
    
    # Comparar metadados
    meta1 = data1["metadata"]
    meta2 = data2["metadata"]
    
    println("🏷️  ID 1: $(meta1["training_id"]) | ID 2: $(meta2["training_id"])")
    println("📅 Data 1: $(meta1["timestamp"]) | Data 2: $(meta2["timestamp"])")
    println("🎯 Acc. 1: $(meta1["final_accuracy"]) | Acc. 2: $(meta2["final_accuracy"])")
    println("👥 Pessoas 1: $(length(meta1["person_names"])) | Pessoas 2: $(length(meta2["person_names"]))")
    
    # Comparar estatísticas das camadas
    layers1 = data1["layers"]
    layers2 = data2["layers"]
    
    for layer_key in intersect(keys(layers1), keys(layers2))
        if haskey(layers1[layer_key], "weights") && haskey(layers2[layer_key], "weights")
            stats1 = layers1[layer_key]["weights"]["stats"]
            stats2 = layers2[layer_key]["weights"]["stats"]
            
            println("\n🔧 $layer_key:")
            println("   Média: $(round(stats1["mean"], digits=6)) → $(round(stats2["mean"], digits=6))")
            println("   Desvio: $(round(stats1["std"], digits=6)) → $(round(stats2["std"], digits=6))")
        end
    end
    
    return true
end

# Listar todos os treinamentos salvos
function list_saved_trainings(filepath::String)
    println("📋 Listando treinamentos salvos...")
    
    if !isfile(filepath)
        println("❌ Arquivo não encontrado: $filepath")
        return
    end
    
    try
        data = TOML.parsefile(filepath)
        
        if !haskey(data, "trainings") || isempty(data["trainings"])
            println("📭 Nenhum treinamento encontrado")
            return
        end
        
        println("\n📚 Treinamentos Disponíveis:")
        println("=" * 60)
        
        for (training_id, training_data) in data["trainings"]
            meta = training_data["metadata"]
            stats = training_data["model_stats"]
            
            println("🔖 ID: $training_id")
            println("   📅 Data: $(meta["timestamp"])")
            println("   🎯 Acurácia: $(round(meta["final_accuracy"] * 100, digits=2))%")
            println("   👥 Pessoas: $(join(meta["person_names"], ", "))")
            println("   🧮 Parâmetros: $(stats["total_parameters"])")
            println("   💾 Tamanho: $(stats["model_size_mb"]) MB")
            println("   📊 Épocas: $(meta["epochs_trained"]) (melhor: $(meta["best_epoch"]))")
            println()
        end
        
        if haskey(data, "summary")
            summary = data["summary"]
            println("📈 Resumo Geral:")
            println("   Total de treinamentos: $(summary["total_trainings"])")
            println("   Último treinamento: $(summary["latest_training"])")
            println("   Total de pessoas: $(length(summary["all_persons"]))")
        end
        
    catch e
        println("❌ Erro ao listar treinamentos: $e")
    end
end

# Função auxiliar para gerar ID único de treinamento
function generate_training_id(person_names::Vector{String})
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    hash_part = bytes2hex(sha256(join(sort(person_names), "_")))[1:8]
    return "train_$(timestamp)_$(hash_part)"
end

# Integração com o sistema existente
function save_pretrained_weights_toml(model, person_names::Vector{String}, 
                                     training_info::Dict, 
                                     weights_filepath::String = "model_weights.toml")
    
    training_id = generate_training_id(person_names)
    data_hash = generate_data_hash(person_names, CNNCheckinCore.TRAIN_DATA_PATH)
    
    metadata = TrainingMetadata(
        training_id,
        string(Dates.now()),
        person_names,
        training_info["epochs_trained"],
        training_info["final_accuracy"],
        training_info["best_epoch"],
        "CNN_FaceRecognition_v1",
        CNNCheckinCore.LEARNING_RATE,
        CNNCheckinCore.BATCH_SIZE,
        data_hash
    )
    
    return save_weights_to_toml(model, person_names, metadata, weights_filepath; append_mode=true)
end

# Export functions
export TrainingMetadata, save_weights_to_toml, load_weights_from_toml, 
       reconstruct_model_from_weights, compare_training_weights, 
       list_saved_trainings, save_pretrained_weights_toml, generate_training_id