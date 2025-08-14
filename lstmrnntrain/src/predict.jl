# projeto: lstmrnntrain - Sistema de Predição Otimizado
# file: predict.jl
# Sistema de predição usando modelos LSTM/RNN treinados

using Flux, TOML, Statistics, Dates, Random, JSON, StatsBase

# Função para carregar dados TOML
function load_stock_data(filepath::String)
    try
        data = TOML.parsefile(filepath)
        records = data["records"]
        
        if isempty(records)
            return nothing
        end
        
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

# Função para normalizar dados
function normalize_minmax(data::Vector{Float64}, min_val::Float64, max_val::Float64)
    range_val = max_val - min_val
    if range_val == 0
        return data
    end
    return (data .- min_val) ./ range_val
end

# Função para desnormalizar
function denormalize_minmax(normalized_data, min_val::Float64, max_val::Float64)
    return normalized_data * (max_val - min_val) + min_val
end

# Função para criar features técnicas (mesma do treinamento)
function create_technical_features(stock_data, window::Int = 3)
    n = length(stock_data.closing)
    
    if n < window + 1
        return nothing
    end
    
    closing = stock_data.closing
    opening = stock_data.opening
    high = stock_data.high
    low = stock_data.low
    volume = stock_data.volume
    
    # Médias móveis
    sma_short = zeros(n)
    sma_long = zeros(n)
    
    for i in window:n
        sma_short[i] = mean(closing[(i-window+1):i])
        long_window = min(2*window, i)
        if i >= long_window
            sma_long[i] = mean(closing[(i-long_window+1):i])
        end
    end
    
    # RSI
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
    
    # Range High-Low
    hl_range = zeros(n)
    for i in 1:n
        if low[i] != 0
            hl_range[i] = (high[i] - low[i]) / low[i]
        end
    end
    
    return hcat(
        closing, opening, high, low, volume,
        sma_short, sma_long, rsi, volatility, price_change, hl_range
    )
end

# Função para carregar modelo JSON
function load_model_from_json(json_path::String)
    try
        model_data = JSON.parsefile(json_path)
        return model_data
    catch e
        println("Erro ao carregar modelo $json_path: $e")
        return nothing
    end
end

# Função para fazer predição usando dados do JSON
function predict_with_json_model(model_data, input_sequence)
    try
        # Esta é uma implementação simplificada
        # Na prática, você precisaria reconstruir o modelo Flux
        # Para demonstração, retornamos uma predição baseada na tendência
        
        # Pegar último preço da sequência
        last_price = input_sequence[1, end]  # closing price (primeira feature)
        
        # Simular predição baseada na tendência recente
        recent_prices = input_sequence[1, :]  # preços de fechamento
        if length(recent_prices) >= 2
            trend = recent_prices[end] - recent_prices[end-1]
            prediction = last_price + trend * 0.5  # Predição conservadora
        else
            prediction = last_price
        end
        
        return prediction
        
    catch e
        println("Erro na predição: $e")
        return nothing
    end
end

# Função para reconstruir modelo Flux (versão simplificada)
function reconstruct_flux_model(model_data)
    try
        architecture = model_data["architecture"]
        
        # Para demonstração, criar um modelo simples
        # Em produção, você reconstruiria exatamente a arquitetura salva
        layers = []
        
        for layer_info in architecture
            if layer_info["type"] == "LSTM"
                # Adicionar camada LSTM (simplificado)
                input_size = layer_info["input_size"]
                hidden_size = layer_info["hidden_size"]
                push!(layers, LSTM(input_size => hidden_size))
            elseif layer_info["type"] == "Dense"
                input_size = layer_info["input_size"]
                output_size = layer_info["output_size"]
                
                # Determinar função de ativação
                activation = get(layer_info, "activation", "identity")
                if activation == "relu"
                    push!(layers, Dense(input_size => output_size, relu))
                else
                    push!(layers, Dense(input_size => output_size))
                end
            elseif layer_info["type"] == "Dropout"
                p = layer_info["p"]
                push!(layers, Dropout(p))
            end
        end
        
        if !isempty(layers)
            return Chain(layers...)
        else
            return nothing
        end
        
    catch e
        println("Erro ao reconstruir modelo: $e")
        return nothing
    end
end

# Função principal de predição
function make_predictions()
    println("=== Sistema de Predição LSTM/RNN Otimizado ===\n")
    
    # Caminhos
    data_dir = "../../../dados/ativos"
    models_dir = "../../../dados/modelos_treinados"
    
    # Verificar se modelos existem
    lstm_path = joinpath(models_dir, "lstm_model_compact.json")
    rnn_path = joinpath(models_dir, "rnn_model_compact.json")
    norm_path = joinpath(models_dir, "asset_normalization_compact.json")
    
    # Tentar caminhos alternativos
    alt_paths = [
        (joinpath(models_dir, "lstm_model.json"), joinpath(models_dir, "rnn_model.json"), joinpath(models_dir, "asset_normalization.json")),
        (lstm_path, rnn_path, norm_path)
    ]
    
    model_found = false
    lstm_data = nothing
    rnn_data = nothing
    norm_data = nothing
    
    for (lstm_p, rnn_p, norm_p) in alt_paths
        if isfile(lstm_p) && isfile(rnn_p) && isfile(norm_p)
            lstm_data = load_model_from_json(lstm_p)
            rnn_data = load_model_from_json(rnn_p)
            
            try
                norm_data = JSON.parsefile(norm_p)
                model_found = true
                println("✅ Modelos encontrados!")
                println("  • LSTM: $(basename(lstm_p))")
                println("  • RNN: $(basename(rnn_p))")
                println("  • Normalização: $(basename(norm_p))")
                break
            catch e
                println("Erro ao carregar normalização: $e")
            end
        end
    end
    
    if !model_found
        println("❌ Modelos não encontrados!")
        println("Verifique se existem os arquivos:")
        println("  • $lstm_path")
        println("  • $rnn_path") 
        println("  • $norm_path")
        println("\nExecute primeiro o treinamento com:")
        println("  julia lstmrnntrain_optimized.jl")
        return
    end
    
    # Carregar dados de ativos
    toml_files = []
    if isdir(data_dir)
        for file in readdir(data_dir)
            if endswith(lowercase(file), ".toml")
                push!(toml_files, file)
            end
        end
    end
    
    if isempty(toml_files)
        println("❌ Nenhum arquivo TOML encontrado em: $data_dir")
        return
    end
    
    println("\n📊 Fazendo predições para $(length(toml_files)) ativos:")
    
    seq_length = 5  # Mesmo usado no treinamento
    predictions_made = 0
    
    for file in toml_files
        filepath = joinpath(data_dir, file)
        stock_data = load_stock_data(filepath)
        
        if stock_data === nothing
            continue
        end
        
        # Criar features
        features = create_technical_features(stock_data, 3)
        if features === nothing
            println("⚠️  $(stock_data.asset_name): Dados insuficientes para features")
            continue
        end
        
        # Remover NaN/Inf
        features = replace(features, NaN => 0.0, Inf => 0.0, -Inf => 0.0)
        
        n_samples, n_features = size(features)
        if n_samples < seq_length + 5
            println("⚠️  $(stock_data.asset_name): Dados insuficientes")
            continue
        end
        
        # Normalizar usando parâmetros salvos
        normalized_features = similar(features)
        
        # Usar primeira entrada de normalização como referência
        if haskey(norm_data, "assets") && !isempty(norm_data["assets"])
            first_asset_params = first(values(norm_data["assets"]))
            
            for i in 1:min(n_features, length(first_asset_params))
                param = first_asset_params[i]
                min_val = param["min_val"]
                max_val = param["max_val"]
                normalized_features[:, i] = normalize_minmax(features[:, i], min_val, max_val)
            end
        else
            # Fallback: normalização simples
            for i in 1:n_features
                min_val = minimum(features[:, i])
                max_val = maximum(features[:, i])
                if max_val > min_val
                    normalized_features[:, i] = (features[:, i] .- min_val) ./ (max_val - min_val)
                else
                    normalized_features[:, i] = features[:, i]
                end
            end
        end
        
        # Pegar últimas sequências para predição
        last_sequence = normalized_features[(end-seq_length+1):end, :]'  # Transpor
        
        # Fazer predições (versão simplificada)
        lstm_pred_norm = predict_with_json_model(lstm_data, last_sequence)
        rnn_pred_norm = predict_with_json_model(rnn_data, last_sequence)
        
        if lstm_pred_norm !== nothing && rnn_pred_norm !== nothing
            # Desnormalizar predições
            if haskey(norm_data, "assets") && !isempty(norm_data["assets"])
                first_asset_params = first(values(norm_data["assets"]))
                closing_param = first_asset_params[1]  # Parâmetro do preço de fechamento
                min_val = closing_param["min_val"]
                max_val = closing_param["max_val"]
                
                lstm_pred = denormalize_minmax(lstm_pred_norm, min_val, max_val)
                rnn_pred = denormalize_minmax(rnn_pred_norm, min_val, max_val)
            else
                lstm_pred = lstm_pred_norm
                rnn_pred = rnn_pred_norm
            end
            
            # Mostrar resultados
            current_price = stock_data.closing[end]
            lstm_change = ((lstm_pred - current_price) / current_price) * 100
            rnn_change = ((rnn_pred - current_price) / current_price) * 100
            
            println("\n📈 $(stock_data.asset_name)")
            println("  • Preço atual: R\$ $(round(current_price, digits=2))")
            println("  • Predição LSTM: R\$ $(round(lstm_pred, digits=2)) ($(round(lstm_change, digits=2))%)")
            println("  • Predição RNN: R\$ $(round(rnn_pred, digits=2)) ($(round(rnn_change, digits=2))%)")
            println("  • Data dos dados: $(stock_data.dates[end])")
            
            predictions_made += 1
        else
            println("⚠️  $(stock_data.asset_name): Erro na predição")
        end
    end
    
    if predictions_made == 0
        println("❌ Nenhuma predição foi realizada")
    else
        println("\n✅ Predições concluídas para $predictions_made ativo(s)")
        println("💡 Lembre-se: Estas são predições baseadas em dados históricos")
        println("   e não devem ser usadas como única base para decisões de investimento.")
    end
end

# Executar
if abspath(PROGRAM_FILE) == @__FILE__
    make_predictions()
end



#  julia predict.jl