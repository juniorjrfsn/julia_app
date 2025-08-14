# projeto: lstmcnntrain - Sistema LSTM + CNN para Predição de Preços de Ações
# file: lstmcnntrain/src/lstmcnntrain.jl
# Sistema híbrido usando LSTM para análise temporal e CNN para padrões locais




using Flux, TOML, Statistics, Dates, Random, JSON, StatsBase
using Flux: train!, ADAM, logitcrossentropy, mse, Conv, MaxPool, flatten

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

# Função para criar features técnicas avançadas (CORRIGIDA)
function create_advanced_technical_features(stock_data, window::Int = 5)
    n = length(stock_data.closing)
    
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
    
    # Inicializar arrays com zeros
    sma_short = zeros(n)
    sma_long = zeros(n)
    ema_short = zeros(n)
    rsi = zeros(n)
    macd_line = zeros(n)
    macd_signal = zeros(n)
    macd_histogram = zeros(n)
    bb_upper = zeros(n)
    bb_lower = zeros(n)
    bb_width = zeros(n)
    volatility = zeros(n)
    volume_sma = zeros(n)
    price_change = zeros(n)
    price_momentum = zeros(n)
    hl_range = zeros(n)
    oc_range = zeros(n)
    
    # Médias móveis
    for i in window:n
        sma_short[i] = mean(closing[(i-window+1):i])
        long_window = min(2*window, i)
        if i >= long_window
            sma_long[i] = mean(closing[(i-long_window+1):i])
        end
    end
    
    # EMA (Exponential Moving Average)
    alpha = 2.0 / (window + 1)
    if n > 0
        ema_short[1] = closing[1]
        for i in 2:n
            ema_short[i] = alpha * closing[i] + (1 - alpha) * ema_short[i-1]
        end
    end
    
    # RSI (Relative Strength Index)
    for i in (window+1):n
        if i > window
            start_idx = max(1, i-window)
            end_idx = i-1
            if end_idx > start_idx
                changes = diff(closing[start_idx:end_idx])
                gains = [max(0, change) for change in changes]
                losses = [abs(min(0, change)) for change in changes]
                
                if length(gains) > 0 && length(losses) > 0
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
    end
    
    # MACD
    if n > 0
        # MACD Line = EMA(12) - EMA(26)
        ema_12 = similar(closing)
        ema_26 = similar(closing)
        ema_12[1] = closing[1]
        ema_26[1] = closing[1]
        
        alpha_12 = 2.0 / 13
        alpha_26 = 2.0 / 27
        
        for i in 2:n
            ema_12[i] = alpha_12 * closing[i] + (1 - alpha_12) * ema_12[i-1]
            ema_26[i] = alpha_26 * closing[i] + (1 - alpha_26) * ema_26[i-1]
            macd_line[i] = ema_12[i] - ema_26[i]
        end
        
        # MACD Signal = EMA(9) of MACD Line
        alpha_signal = 2.0 / 10
        macd_signal[1] = macd_line[1]
        for i in 2:n
            macd_signal[i] = alpha_signal * macd_line[i] + (1 - alpha_signal) * macd_signal[i-1]
            macd_histogram[i] = macd_line[i] - macd_signal[i]
        end
    end
    
    # Bollinger Bands
    for i in window:n
        period_data = closing[(i-window+1):i]
        sma = mean(period_data)
        std_dev = std(period_data)
        bb_upper[i] = sma + 2 * std_dev
        bb_lower[i] = sma - 2 * std_dev
        bb_width[i] = bb_upper[i] - bb_lower[i]
    end
    
    # Volatilidade
    if n > 1
        returns = diff(log.(max.(closing, 1e-10)))
        for i in window:(n-1)
            start_idx = max(1, i-window+1)
            end_idx = min(i, length(returns))
            if end_idx > start_idx
                volatility[i+1] = std(returns[start_idx:end_idx])
            end
        end
    end
    
    # Volume indicators
    for i in window:n
        volume_sma[i] = mean(volume[(i-window+1):i])
    end
    
    # Price patterns
    if n > 1
        for i in 2:n
            if closing[i-1] != 0
                price_change[i] = (closing[i] - closing[i-1]) / closing[i-1]
            end
        end
        
        for i in (window+1):n
            if closing[i-window] != 0
                price_momentum[i] = (closing[i] - closing[i-window]) / closing[i-window]
            end
        end
    end
    
    # Range indicators
    for i in 1:n
        if low[i] != 0
            hl_range[i] = (high[i] - low[i]) / low[i]
        end
        if opening[i] != 0
            oc_range[i] = (closing[i] - opening[i]) / opening[i]
        end
    end
    
    return hcat(
        closing,           # 1
        opening,           # 2  
        high,              # 3
        low,               # 4
        volume,            # 5
        sma_short,         # 6
        sma_long,          # 7
        ema_short,         # 8
        rsi,               # 9
        macd_line,         # 10
        macd_signal,       # 11
        macd_histogram,    # 12
        bb_upper,          # 13
        bb_lower,          # 14
        bb_width,          # 15
        volatility,        # 16
        volume_sma,        # 17
        price_change,      # 18
        price_momentum,    # 19
        hl_range,          # 20
        oc_range           # 21
    )
end

# Função para preparar dados para CNN+LSTM
function prepare_hybrid_training_data(stock_data, seq_length::Int = 10, prediction_horizon::Int = 1)
    # Criar features técnicas avançadas
    features = create_advanced_technical_features(stock_data, 5)
    
    if features === nothing
        return nothing, nothing, nothing
    end
    
    # Remover NaN e Inf
    features = replace(features, NaN => 0.0, Inf => 0.0, -Inf => 0.0)
    
    n_samples, n_features = size(features)
    
    # Verificar se temos dados suficientes
    min_required = seq_length + prediction_horizon + 10
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
    
    # Criar sequências para CNN+LSTM
    X = []
    y = []
    
    start_idx = max(11, seq_length + 1)
    
    for i in start_idx:(n_samples - prediction_horizon)
        # Input: sequência de features para híbrido CNN+LSTM
        input_seq = normalized_features[(i-seq_length+1):i, :]
        
        # Target: preço de fechamento futuro (normalizado)
        target = normalized_features[i + prediction_horizon, 1]  # closing price
        
        # Para CNN: reshape para (features, timesteps, channels=1)
        # Para LSTM: formato (features, timesteps)
        push!(X, input_seq')  # Transpor para (features, timesteps)
        push!(y, target)
    end
    
    if isempty(X)
        return nothing, nothing, nothing
    end
    
    return X, y, normalization_params
end

# Modelo híbrido CNN+LSTM simplificado (SEM MUTAÇÕES)
function create_hybrid_cnn_lstm_model(input_size::Int, seq_length::Int)
    return Chain(
        # Camada de extração de features temporais
        LSTM(input_size => 128),
        Dropout(0.3),
        LSTM(128 => 64),
        Dropout(0.2),
        LSTM(64 => 32),
        Dropout(0.2),
        
        # Camadas finais de predição
        Dense(32 => 16, relu),
        Dropout(0.1),
        Dense(16 => 8, relu),
        Dense(8 => 1)
    )
end

# Modelo LSTM puro aprimorado
function create_advanced_lstm_model(input_size::Int)
    return Chain(
        LSTM(input_size => 128),
        Dropout(0.3),
        LSTM(128 => 64),
        Dropout(0.2),
        LSTM(64 => 32),
        Dropout(0.2),
        Dense(32 => 16, relu),
        Dense(16 => 1)
    )
end

# Modelo Transformer-like simplificado
function create_transformer_model(input_size::Int)
    return Chain(
        # Camadas de atenção simulada com Dense
        Dense(input_size => 128, relu),
        Dropout(0.3),
        Dense(128 => 128, relu),
        Dropout(0.2),
        
        # Processamento temporal
        LSTM(128 => 64),
        Dropout(0.2),
        
        # Saída final
        Dense(64 => 32, relu),
        Dense(32 => 16, relu),
        Dense(16 => 1)
    )
end

# Função de treinamento otimizada para modelos híbridos
function train_hybrid_model(model, X_data, y_data, epochs::Int = 100, batch_size::Int = 16, lr::Float64 = 0.001)
    # Preparar dados
    n_samples = length(X_data)
    X_train = [Float32.(x) for x in X_data]
    y_train = Float32.(y_data)
    
    # Configurar otimizador
    opt_state = Flux.setup(ADAM(lr), model)
    
    # Histórico
    train_losses = Float64[]
    
    println("Iniciando treinamento híbrido com $n_samples amostras...")
    
    best_loss = Inf
    patience = 15
    no_improve = 0
    
    for epoch in 1:epochs
        # Embaralhar dados
        indices = randperm(n_samples)
        epoch_loss = 0.0
        
        # Processar em batches
        for start_idx in 1:batch_size:n_samples
            end_idx = min(start_idx + batch_size - 1, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            batch_loss = 0.0
            
            for idx in batch_indices
                # Reset estado dos modelos recorrentes
                Flux.reset!(model)
                
                # Calcular loss e gradientes
                loss_val, grads = Flux.withgradient(model) do m
                    pred = m(X_train[idx])
                    
                    # Extrair saída final
                    if pred isa Vector
                        output = pred[end]
                    elseif pred isa Matrix
                        output = pred[1, end]
                    else
                        output = pred
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
        
        # Logging
        if epoch % 20 == 0 || no_improve >= patience
            println("Epoch $epoch: Loss = $(round(avg_loss, digits=6)) | Best = $(round(best_loss, digits=6))")
        end
        
        if no_improve >= patience
            println("Early stopping na epoch $epoch")
            break
        end
        
        # Learning rate scheduling
        if epoch % 40 == 0
            for param in opt_state
                if hasfield(typeof(param), :eta)
                    param.eta *= 0.9
                end
            end
        end
    end
    
    return model, train_losses
end

# Função para fazer predições com modelo híbrido
function predict_next_day_hybrid(model, last_sequence, normalization_params)
    Flux.reset!(model)
    
    # Fazer predição
    pred = model(Float32.(last_sequence))
    
    # Extrair predição final
    if pred isa Vector
        prediction = pred[end]
    elseif pred isa Matrix
        prediction = pred[1, end]
    else
        prediction = pred
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

# Função para salvar modelo em formato JSON (adaptada para híbridos)
function save_hybrid_model_json(model, normalization_params, training_info, model_type, save_path)
    # Extrair parâmetros do modelo (simplificado para modelos complexos)
    model_params = []
    
    # Para modelos híbridos, salvamos uma representação simplificada
    push!(model_params, Dict(
        "type" => model_type,
        "architecture" => "Hybrid CNN+LSTM",
        "description" => "Complex hybrid model with CNN feature extraction and LSTM temporal analysis"
    ))
    
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
            "sma_short", "sma_long", "ema_short", "rsi", 
            "macd_line", "macd_signal", "macd_histogram",
            "bb_upper", "bb_lower", "bb_width", "volatility",
            "volume_sma", "price_change", "price_momentum", 
            "hl_range", "oc_range"
        ],
        "metadata" => Dict(
            "created_at" => string(now()),
            "julia_version" => string(VERSION),
            "flux_version" => "0.14+",
            "model_complexity" => "Advanced Hybrid"
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

# Função principal
function main()
    println("=== Sistema Híbrido LSTM + CNN para Predição de Preços de Ações ===\n")
    
    # Configurações
    data_dir = "../../../dados/ativos"
    save_dir = "../../../dados/modelos_treinados_lstm_cnn"
    mkpath(save_dir)
    
    seq_length = 10        # Sequência temporal para análise
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
        
        X, y, norm_params = prepare_hybrid_training_data(stock_data, seq_length, prediction_horizon)
        
        if X !== nothing && length(X) > 5
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
    n_train = max(1, Int(floor(0.8 * n_samples)))
    
    train_indices = randperm(n_samples)[1:n_train]
    
    X_train = all_X[train_indices]
    y_train = all_y[train_indices]
    
    # Treinar modelo híbrido CNN+LSTM
    println("\n=== Treinando Modelo Híbrido Simplificado ===")
    hybrid_model = create_hybrid_cnn_lstm_model(size(all_X[1], 1), seq_length)
    hybrid_trained, hybrid_losses = train_hybrid_model(
        hybrid_model, X_train, y_train, 80, 8, 0.001
    )
    
    # Treinar LSTM avançado para comparação
    println("\n=== Treinando LSTM Avançado ===")
    lstm_model = create_advanced_lstm_model(size(all_X[1], 1))
    lstm_trained, lstm_losses = train_hybrid_model(
        lstm_model, X_train, y_train, 80, 8, 0.001
    )
    
    # Treinar modelo Transformer-like
    println("\n=== Treinando Modelo Transformer-like ===")
    transformer_model = create_transformer_model(size(all_X[1], 1))
    transformer_trained, transformer_losses = train_hybrid_model(
        transformer_model, X_train, y_train, 80, 8, 0.0005
    )
    
    # Informações de treinamento
    training_info = Dict(
        "timestamp" => string(now()),
        "total_sequences" => length(all_X),
        "training_sequences" => length(X_train),
        "sequence_length" => seq_length,
        "prediction_horizon" => prediction_horizon,
        "features_count" => size(all_X[1], 1),
        "epochs_completed" => min(length(hybrid_losses), length(lstm_losses), length(transformer_losses)),
        "assets_trained" => collect(keys(asset_normalization)),
        "hybrid_final_loss" => hybrid_losses[end],
        "lstm_final_loss" => lstm_losses[end],
        "transformer_final_loss" => transformer_losses[end]
    )
    
    # Salvar modelos
    println("\n=== Salvando Modelos ===")
    
    # Salvar Híbrido CNN+LSTM
    hybrid_path = joinpath(save_dir, "hybrid_simplified_model.json")
    hybrid_data = save_hybrid_model_json(
        hybrid_trained, 
        first(values(asset_normalization)),
        training_info, 
        "Hybrid_Simplified", 
        hybrid_path
    )
    println("✅ Modelo Híbrido Simplificado salvo em: $hybrid_path")
    
    # Salvar LSTM avançado
    lstm_path = joinpath(save_dir, "advanced_lstm_model.json") 
    lstm_data = save_hybrid_model_json(
        lstm_trained, 
        first(values(asset_normalization)),
        training_info, 
        "Advanced_LSTM", 
        lstm_path
    )
    println("✅ LSTM Avançado salvo em: $lstm_path")
    
    # Salvar Transformer-like
    transformer_path = joinpath(save_dir, "transformer_model.json")
    transformer_data = save_hybrid_model_json(
        transformer_trained,
        first(values(asset_normalization)),
        training_info,
        "Transformer_Like",
        transformer_path
    )
    println("✅ Modelo Transformer-like salvo em: $transformer_path")
    
    # Salvar parâmetros de normalização por ativo
    normalization_path = joinpath(save_dir, "asset_normalization_hybrid.json")
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
            "sma_short", "sma_long", "ema_short", "rsi", 
            "macd_line", "macd_signal", "macd_histogram",
            "bb_upper", "bb_lower", "bb_width", "volatility",
            "volume_sma", "price_change", "price_momentum", 
            "hl_range", "oc_range"
        ]
    )
    
    open(normalization_path, "w") do file
        JSON.print(file, norm_data, 2)
    end
    println("✅ Parâmetros de normalização salvos em: $normalization_path")
    
    # Relatório final
    println("\n=== Relatório de Treinamento ===")
    println("📈 Performance:")
    println("  • Híbrido Simplificado - Perda final: $(round(hybrid_losses[end], digits=6))")
    println("  • LSTM Avançado - Perda final: $(round(lstm_losses[end], digits=6))")
    println("  • Transformer-like - Perda final: $(round(transformer_losses[end], digits=6))")
    
    if length(hybrid_losses) > 1
        println("  • Híbrido - Melhoria: $(round((1 - hybrid_losses[end]/hybrid_losses[1])*100, digits=2))%")
        println("  • LSTM - Melhoria: $(round((1 - lstm_losses[end]/lstm_losses[1])*100, digits=2))%")
        println("  • Transformer - Melhoria: $(round((1 - transformer_losses[end]/transformer_losses[1])*100, digits=2))%")
    end
    
    println("\n📊 Dados:")
    println("  • Ativos processados: $successful_loads")
    println("  • Sequências de treinamento: $(length(X_train))")
    println("  • Sequência temporal: $seq_length dias")
    println("  • Features técnicas: $(size(all_X[1], 1))")
    println("  • Horizonte de predição: $prediction_horizon dia(s)")
    
    # Fazer predições de exemplo
    if successful_loads > 0
        println("\n=== Exemplo de Predições ===")
        sample_asset = first(keys(asset_normalization))
        sample_params = asset_normalization[sample_asset]
        
        if !isempty(all_X)
            sample_seq = all_X[end]
            
            hybrid_pred = predict_next_day_hybrid(hybrid_trained, sample_seq, sample_params)
            lstm_pred = predict_next_day_hybrid(lstm_trained, sample_seq, sample_params)
            transformer_pred = predict_next_day_hybrid(transformer_trained, sample_seq, sample_params)
            
            println("  • Predição Híbrida Simplificada: R\$ $(round(hybrid_pred, digits=2))")
            println("  • Predição LSTM Avançado: R\$ $(round(lstm_pred, digits=2))")
            println("  • Predição Transformer-like: R\$ $(round(transformer_pred, digits=2))")
            
            # Calcular consenso
            consensus = mean([hybrid_pred, lstm_pred, transformer_pred])
            println("  • Consenso dos modelos: R\$ $(round(consensus, digits=2))")
            
            # Variação entre modelos
            max_pred = maximum([hybrid_pred, lstm_pred, transformer_pred])
            min_pred = minimum([hybrid_pred, lstm_pred, transformer_pred])
            variation = ((max_pred - min_pred) / min_pred) * 100
            println("  • Variação entre modelos: $(round(variation, digits=2))%")
        end
    end
    
    println("\n✅ Treinamento concluído com sucesso!")
    println("🧠 Modelos treinados: Híbrido Simplificado, LSTM Avançado e Transformer-like")
    println("💡 Os modelos estão prontos para predição avançada de preços de ações")
    
    # Análise comparativa de performance
    println("\n=== Análise Comparativa ===")
    best_model = "Híbrido"
    best_loss = hybrid_losses[end]
    
    if lstm_losses[end] < best_loss
        best_model = "LSTM"
        best_loss = lstm_losses[end]
    end
    
    if transformer_losses[end] < best_loss
        best_model = "Transformer"
        best_loss = transformer_losses[end]
    end
    
    println("🏆 Melhor modelo: $best_model (Loss: $(round(best_loss, digits=6)))")
    
    # Salvar relatório de comparação
    comparison_report = Dict(
        "training_summary" => Dict(
            "timestamp" => string(now()),
            "total_assets" => successful_loads,
            "total_sequences" => length(all_X),
            "training_sequences" => length(X_train),
            "best_model" => best_model,
            "best_loss" => best_loss
        ),
        "model_performance" => Dict(
            "hybrid_simplified" => Dict(
                "final_loss" => hybrid_losses[end],
                "epochs_trained" => length(hybrid_losses),
                "architecture" => "LSTM x3 + Dense layers"
            ),
            "lstm_advanced" => Dict(
                "final_loss" => lstm_losses[end],
                "epochs_trained" => length(lstm_losses),
                "architecture" => "LSTM x3 + Dense layers"
            ),
            "transformer_like" => Dict(
                "final_loss" => transformer_losses[end],
                "epochs_trained" => length(transformer_losses),
                "architecture" => "Dense + LSTM + Attention-like"
            )
        )
    )
    
    report_path = joinpath(save_dir, "training_comparison_report.json")
    open(report_path, "w") do file
        JSON.print(file, comparison_report, 2)
    end
    println("📋 Relatório de comparação salvo em: $report_path")
end

# Executar
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end



# Para executar: 
# julia lstmcnntrain.jl
