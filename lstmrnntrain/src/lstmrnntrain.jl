#module lstmrnntrain
#greet() = print("Hello World!")
#end # module lstmrnntrain

 
 
# projeto: lstmrnntrain - Versão Otimizada para Datasets Pequenos
# file: lstmrnntrain/src/lstmrnntrain.jl
# Sistema de treinamento LSTM/RNN para predição de preços de ações

using Flux, TOML, Statistics, Dates, Random, JSON, StatsBase
using Flux: train!, ADAM, logitcrossentropy, mse

# Configurar seed para reprodutibilidade
Random.seed!(42)

# Função para carregar dados TOML
function load_stock_data(filepath::String)
    try
        data = TOML.parsefile(filepath)
        records = data["records"]
        
        if isempty(records)
            return nothing
        end
        
        # Extrair dados relevantes e converter para Float64
        dates = [record["date"] for record in records]
        closing = Float64.([record["closing"] for record in records])
        volume = Float64.([record["volume"] for record in records])
        variation = Float64.([record["variation"] for record in records])
        opening = Float64.([record["opening"] for record in records])
        high = Float64.([record["high"] for record in records])
        low = Float64.([record["low"] for record in records])
        
        return (
            dates = dates,
            closing = closing,
            opening = opening,
            high = high,
            low = low,
            volume = volume,
            variation = variation,
            asset_name = data["asset"]
        )
    catch e
        println("Erro ao carregar $filepath: $e")
        return nothing
    end
end

# Função para normalizar dados usando Min-Max normalization
function normalize_minmax(data::Vector{Float64})
    min_val = minimum(data)
    max_val = maximum(data)
    range_val = max_val - min_val
    
    if range_val == 0
        return data, min_val, max_val
    end
    
    normalized = (data .- min_val) ./ range_val
    return normalized, min_val, max_val
end

# Função para desnormalizar
function denormalize_minmax(normalized_data, min_val::Float64, max_val::Float64)
    return normalized_data * (max_val - min_val) + min_val
end

# Função para criar features técnicas otimizada para datasets pequenos
function create_technical_features(stock_data, window::Int = 3)  # Reduzido de 5 para 3
    n = length(stock_data.closing)
    
    # Verificar se temos dados suficientes
    if n < window + 1
        println("Aviso: Dados insuficientes para calcular features técnicas (n=$n, window=$window)")
        return nothing
    end
    
    # Features básicas
    closing = stock_data.closing
    opening = stock_data.opening
    high = stock_data.high
    low = stock_data.low
    volume = stock_data.volume
    
    # Médias móveis com janela menor
    sma_short = zeros(n)
    sma_long = zeros(n)
    
    for i in window:n
        sma_short[i] = mean(closing[(i-window+1):i])
        long_window = min(2*window, i)  # Janela adaptativa
        if i >= long_window
            sma_long[i] = mean(closing[(i-long_window+1):i])
        end
    end
    
    # RSI simplificado - versão corrigida
    rsi = zeros(n)
    for i in (window+1):n
        if i > window
            start_idx = max(1, i-window)
            end_idx = i-1
            if end_idx > start_idx
                changes = diff(closing[start_idx:end_idx])
                gains = [max(0, change) for change in changes]
                losses = [abs(min(0, change)) for change in changes]
                
                avg_gain = mean(gains)
                avg_loss = mean(losses)
                
                if avg_loss == 0
                    rsi[i] = 100
                else
                    rs = avg_gain / avg_loss
                    rsi[i] = 100 - (100 / (1 + rs))
                end
            end
        end
    end
    
    # Volatilidade
    volatility = zeros(n)
    if n > 1
        returns = diff(log.(max.(closing, 1e-10)))
        for i in window:n-1
            start_idx = max(1, i-window+1)
            end_idx = min(i, length(returns))
            if end_idx > start_idx
                volatility[i+1] = std(returns[start_idx:end_idx])
            end
        end
    end
    
    # Retornos percentuais
    price_change = zeros(n)
    if n > 1
        for i in 2:n
            if closing[i-1] != 0
                price_change[i] = (closing[i] - closing[i-1]) / closing[i-1]
            end
        end
    end
    
    # Range High-Low normalizado
    hl_range = zeros(n)
    for i in 1:n
        if low[i] != 0
            hl_range[i] = (high[i] - low[i]) / low[i]
        end
    end
    
    return hcat(
        closing,                   # 1
        opening,                   # 2  
        high,                      # 3
        low,                       # 4
        volume,                    # 5
        sma_short,                 # 6
        sma_long,                  # 7
        rsi,                       # 8
        volatility,                # 9
        price_change,              # 10
        hl_range                   # 11
    )
end

# Função para preparar dados para treinamento - otimizada para datasets pequenos
function prepare_training_data(stock_data, seq_length::Int = 5, prediction_horizon::Int = 1)  # Reduzido de 10 para 5
    # Criar features técnicas
    features = create_technical_features(stock_data, 3)
    
    if features === nothing
        return nothing, nothing, nothing
    end
    
    # Remover NaN e Inf
    features = replace(features, NaN => 0.0, Inf => 0.0, -Inf => 0.0)
    
    n_samples, n_features = size(features)
    
    # Verificar se temos dados suficientes - critério mais flexível
    min_required = seq_length + prediction_horizon + 5  # Reduzido de 10 para 5
    if n_samples < min_required
        println("Aviso: $(stock_data.asset_name) - Dados insuficientes: $n_samples < $min_required")
        return nothing, nothing, nothing
    end
    
    # Normalizar cada feature
    normalized_features = similar(features)
    normalization_params = []
    
    for i in 1:n_features
        normalized_features[:, i], min_val, max_val = normalize_minmax(features[:, i])
        push!(normalization_params, (min_val = min_val, max_val = max_val))
    end
    
    # Criar sequências
    X = []
    y = []
    
    start_idx = max(5, seq_length + 1)  # Reduzido de 11 para 5
    
    for i in start_idx:(n_samples - prediction_horizon)
        # Input: sequência de features
        input_seq = normalized_features[(i-seq_length+1):i, :]
        
        # Target: preço de fechamento futuro (normalizado)
        target = normalized_features[i + prediction_horizon, 1]  # closing price
        
        push!(X, input_seq')  # Transpor para (features, timesteps)
        push!(y, target)
    end
    
    if isempty(X)
        return nothing, nothing, nothing
    end
    
    return X, y, normalization_params
end

# Modelo LSTM compacto para datasets pequenos
function create_compact_lstm_model(input_size::Int)
    return Chain(
        LSTM(input_size => 32),      # Reduzido de 64 para 32
        Dropout(0.2),                # Reduzido dropout
        LSTM(32 => 16),              # Reduzido de 32 para 16
        Dropout(0.1),
        Dense(16 => 8, relu),        # Camada intermediária menor
        Dense(8 => 1)                # Output
    )
end

# Modelo RNN compacto
function create_compact_rnn_model(input_size::Int)
    return Chain(
        RNN(input_size => 32, tanh),
        Dropout(0.2),
        RNN(32 => 16, tanh), 
        Dropout(0.1),
        Dense(16 => 8, relu),
        Dense(8 => 1)
    )
end

# Função de treinamento otimizada
function train_model_compact(model, X_data, y_data, epochs::Int = 100, batch_size::Int = 8, lr::Float64 = 0.01)
    # Preparar dados
    n_samples = length(X_data)
    X_train = [Float32.(x) for x in X_data]
    y_train = Float32.(y_data)
    
    # Configurar otimizador
    opt_state = Flux.setup(ADAM(lr), model)
    
    # Histórico
    train_losses = Float64[]
    
    println("Iniciando treinamento com $n_samples amostras...")
    
    best_loss = Inf
    patience = 15  # Reduzido
    no_improve = 0
    
    for epoch in 1:epochs
        # Embaralhar dados
        indices = randperm(n_samples)
        epoch_loss = 0.0
        
        # Processar em batches menores
        for start_idx in 1:batch_size:n_samples
            end_idx = min(start_idx + batch_size - 1, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            batch_loss = 0.0
            
            for idx in batch_indices
                # Reset estado do modelo
                Flux.reset!(model)
                
                # Calcular loss e gradientes
                loss_val, grads = Flux.withgradient(model) do m
                    pred = m(X_train[idx])
                    # Pegar última saída da sequência
                    if pred isa Vector
                        output = pred[end]
                    else
                        output = pred[1, end]
                    end
                    return mse(output, y_train[idx])
                end
                
                # Atualizar gradientes
                if !isnothing(grads[1])
                    Flux.update!(opt_state, model, grads[1])
                end
                
                batch_loss += loss_val
            end
            
            epoch_loss += batch_loss
        end
        
        avg_loss = epoch_loss / n_samples
        push!(train_losses, avg_loss)
        
        # Early stopping
        if avg_loss < best_loss
            best_loss = avg_loss
            no_improve = 0
        else
            no_improve += 1
        end
        
        if epoch % 20 == 0 || no_improve >= patience
            println("Epoch $epoch: Loss = $(round(avg_loss, digits=6))")
        end
        
        if no_improve >= patience
            println("Early stopping na epoch $epoch")
            break
        end
        
        # Learning rate decay
        if epoch % 30 == 0
            for param in opt_state
                if hasfield(typeof(param), :eta)
                    param.eta *= 0.9
                end
            end
        end
    end
    
    return model, train_losses
end

# Função para fazer predições
function predict_next_day(model, last_sequence, normalization_params)
    Flux.reset!(model)
    
    # Fazer predição
    pred = model(Float32.(last_sequence))
    
    if pred isa Vector
        prediction = pred[end]
    else
        prediction = pred[1, end]
    end
    
    # Desnormalizar (usando parâmetros do preço de fechamento - índice 1)
    closing_params = normalization_params[1]
    prediction_denorm = denormalize_minmax(
        prediction, 
        closing_params.min_val, 
        closing_params.max_val
    )
    
    return Float64(prediction_denorm)
end

# Função para salvar modelo em formato JSON
function save_model_json(model, normalization_params, training_info, model_type, save_path)
    # Extrair parâmetros do modelo
    model_params = []
    
    for layer in model.layers
        if layer isa Dense
            push!(model_params, Dict(
                "type" => "Dense",
                "input_size" => size(layer.weight, 2),
                "output_size" => size(layer.weight, 1),
                "weight" => Array(layer.weight),
                "bias" => Array(layer.bias),
                "activation" => string(layer.σ)
            ))
        elseif layer isa LSTM
            cell = layer.cell
            push!(model_params, Dict(
                "type" => "LSTM",
                "input_size" => size(cell.Wi, 2),
                "hidden_size" => size(cell.Wh, 2),
                "Wi" => Array(cell.Wi),
                "Wh" => Array(cell.Wh),
                "bias" => hasfield(typeof(cell), :b) ? Array(cell.b) : nothing
            ))
        elseif layer isa RNN
            cell = layer.cell
            push!(model_params, Dict(
                "type" => "RNN", 
                "input_size" => size(cell.Wi, 2),
                "hidden_size" => size(cell.Wh, 2),
                "Wi" => Array(cell.Wi),
                "Wh" => Array(cell.Wh),
                "bias" => hasfield(typeof(cell), :b) ? Array(cell.b) : nothing,
                "activation" => string(cell.σ)
            ))
        elseif layer isa Dropout
            push!(model_params, Dict(
                "type" => "Dropout",
                "p" => layer.p
            ))
        end
    end
    
    # Preparar dados para salvar
    model_data = Dict(
        "model_type" => model_type,
        "architecture" => model_params,
        "normalization_params" => [
            Dict("min_val" => p.min_val, "max_val" => p.max_val) 
            for p in normalization_params
        ],
        "training_info" => training_info,
        "feature_names" => [
            "closing", "opening", "high", "low", "volume",
            "sma_short", "sma_long", "rsi", "volatility", "price_change", "hl_range"
        ],
        "metadata" => Dict(
            "created_at" => string(now()),
            "julia_version" => string(VERSION),
            "flux_version" => "0.14+"
        )
    )
    
    # Salvar arquivo
    open(save_path, "w") do file
        JSON.print(file, model_data, 2)
    end
    
    return model_data
end

# Função para descobrir arquivos TOML
function find_toml_files(data_dir::String)
    toml_files = []
    
    if isdir(data_dir)
        for file in readdir(data_dir)
            if endswith(lowercase(file), ".toml")
                push!(toml_files, file)
            end
        end
    else
        println("Diretório não encontrado: $data_dir")
    end
    
    return sort(toml_files)
end

# Função principal otimizada
function main()
    println("=== Sistema de Treinamento LSTM/RNN Otimizado para Datasets Pequenos ===\n")
    
    # Configurações otimizadas
    data_dir = "../../../dados/ativos"
    save_dir = "../../../dados/modelos_treinados"
    mkpath(save_dir)
    
    seq_length = 5         # Reduzido de 15 para 5
    prediction_horizon = 1  # Predizer 1 dia à frente
    
    # Encontrar arquivos TOML
    files = find_toml_files(data_dir)
    
    if isempty(files)
        println("❌ Nenhum arquivo TOML encontrado em: $data_dir")
        return
    end
    
    println("📂 Encontrados $(length(files)) arquivos TOML")
    
    # Carregar e processar dados
    all_X = []
    all_y = []
    asset_normalization = Dict()
    successful_loads = 0
    
    for file in files
        filepath = joinpath(data_dir, file)
        stock_data = load_stock_data(filepath)
        
        if stock_data === nothing
            continue
        end
        
        X, y, norm_params = prepare_training_data(stock_data, seq_length, prediction_horizon)
        
        if X !== nothing && length(X) > 3  # Critério mais flexível: pelo menos 3 amostras
            append!(all_X, X)
            append!(all_y, y)
            
            # Salvar parâmetros de normalização por ativo
            asset_normalization[stock_data.asset_name] = norm_params
            
            successful_loads += 1
            println("✅ $(stock_data.asset_name): $(length(X)) sequências criadas")
        else
            println("⚠️  $(stock_data.asset_name): Dados insuficientes")
        end
    end
    
    if isempty(all_X)
        println("❌ Nenhum dado válido para treinamento")
        return
    end
    
    println("\n📊 Dados preparados:")
    println("  • Total de sequências: $(length(all_X))")
    println("  • Ativos processados: $successful_loads")
    println("  • Comprimento da sequência: $seq_length")
    println("  • Features por timestep: $(size(all_X[1], 1))")
    
    # Dividir em treino/validação
    n_samples = length(all_X)
    n_train = max(1, Int(floor(0.8 * n_samples)))  # Garantir pelo menos 1 amostra
    
    train_indices = randperm(n_samples)[1:n_train]
    
    X_train = all_X[train_indices]
    y_train = all_y[train_indices]
    
    # Treinar LSTM
    println("\n=== Treinando Modelo LSTM Compacto ===")
    lstm_model = create_compact_lstm_model(size(all_X[1], 1))
    lstm_trained, lstm_losses = train_model_compact(
        lstm_model, X_train, y_train, 80, 4, 0.01
    )
    
    # Treinar RNN
    println("\n=== Treinando Modelo RNN Compacto ===")
    rnn_model = create_compact_rnn_model(size(all_X[1], 1))
    rnn_trained, rnn_losses = train_model_compact(
        rnn_model, X_train, y_train, 80, 4, 0.01
    )
    
    # Informações de treinamento
    training_info = Dict(
        "timestamp" => string(now()),
        "total_sequences" => length(all_X),
        "training_sequences" => length(X_train),
        "sequence_length" => seq_length,
        "prediction_horizon" => prediction_horizon,
        "features_count" => size(all_X[1], 1),
        "epochs_completed" => min(length(lstm_losses), length(rnn_losses)),
        "assets_trained" => collect(keys(asset_normalization)),
        "lstm_final_loss" => lstm_losses[end],
        "rnn_final_loss" => rnn_losses[end]
    )
    
    # Salvar modelos
    println("\n=== Salvando Modelos ===")
    
    # Salvar LSTM
    lstm_path = joinpath(save_dir, "lstm_model_compact.json")
    lstm_data = save_model_json(
        lstm_trained, 
        first(values(asset_normalization)),
        training_info, 
        "LSTM_Compact", 
        lstm_path
    )
    println("✅ LSTM salvo em: $lstm_path")
    
    # Salvar RNN
    rnn_path = joinpath(save_dir, "rnn_model_compact.json") 
    rnn_data = save_model_json(
        rnn_trained, 
        first(values(asset_normalization)),
        training_info, 
        "RNN_Compact", 
        rnn_path
    )
    println("✅ RNN salvo em: $rnn_path")
    
    # Salvar parâmetros de normalização por ativo
    normalization_path = joinpath(save_dir, "asset_normalization_compact.json")
    norm_data = Dict(
        "assets" => Dict(
            asset => [
                Dict("min_val" => p.min_val, "max_val" => p.max_val) 
                for p in params
            ]
            for (asset, params) in asset_normalization
        ),
        "feature_names" => [
            "closing", "opening", "high", "low", "volume",
            "sma_short", "sma_long", "rsi", "volatility", "price_change", "hl_range"
        ]
    )
    
    open(normalization_path, "w") do file
        JSON.print(file, norm_data, 2)
    end
    println("✅ Parâmetros de normalização salvos em: $normalization_path")
    
    # Relatório final
    println("\n=== Relatório de Treinamento ===")
    println("📈 Performance:")
    println("  • LSTM - Perda final: $(round(lstm_losses[end], digits=6))")
    println("  • RNN - Perda final: $(round(rnn_losses[end], digits=6))")
    if length(lstm_losses) > 1
        println("  • LSTM - Melhoria: $(round((1 - lstm_losses[end]/lstm_losses[1])*100, digits=2))%")
        println("  • RNN - Melhoria: $(round((1 - rnn_losses[end]/rnn_losses[1])*100, digits=2))%")
    end
    
    println("\n📊 Dados:")
    println("  • Ativos processados: $successful_loads")
    println("  • Sequências de treinamento: $(length(X_train))")
    println("  • Sequência temporal: $seq_length dias")
    println("  • Horizonte de predição: $prediction_horizon dia(s)")
    
    println("\n✅ Treinamento concluído com sucesso!")
    println("💡 Os modelos estão prontos para fazer predições do próximo dia de negociação.")
    
    # Fazer predições de exemplo
    if successful_loads > 0
        println("\n=== Exemplo de Predições ===")
        sample_asset = first(keys(asset_normalization))
        sample_params = asset_normalization[sample_asset]
        
        if !isempty(all_X)
            sample_seq = all_X[end]
            
            lstm_pred = predict_next_day(lstm_trained, sample_seq, sample_params)
            rnn_pred = predict_next_day(rnn_trained, sample_seq, sample_params)
            
            println("  • Predição LSTM: R\$ $(round(lstm_pred, digits=2))")
            println("  • Predição RNN: R\$ $(round(rnn_pred, digits=2))")
        end
    end
end

# Executar
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end


# Para executar: 
#  julia lstmrnntrain.jl
 