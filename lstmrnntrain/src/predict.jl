# projeto : lstmrnntrain
# file : lstmrnntrain/src/predict.jl
using Flux, BSON, Statistics, TOML

# Estrutura para armazenar modelos carregados
struct TrainedModels
    lstm_model::Any
    rnn_model::Any
    normalization_params::Any
    seq_length::Int
end

# Função para carregar modelos treinados
function load_trained_models(models_dir::String = "modelos_treinados")
    println("Carregando modelos treinados de: $models_dir")
    
    # Carregar modelo LSTM
    lstm_path = joinpath(models_dir, "lstm_model.bson")
    if isfile(lstm_path)
        BSON.@load lstm_path model
        lstm_model = model
        println("✓ Modelo LSTM carregado")
    else
        error("Arquivo do modelo LSTM não encontrado: $lstm_path")
    end
    
    # Carregar modelo RNN
    rnn_path = joinpath(models_dir, "rnn_model.bson")
    if isfile(rnn_path)
        BSON.@load rnn_path model
        rnn_model = model
        println("✓ Modelo RNN carregado")
    else
        error("Arquivo do modelo RNN não encontrado: $rnn_path")
    end
    
    # Carregar parâmetros de normalização
    params_path = joinpath(models_dir, "normalization_params.bson")
    if isfile(params_path)
        BSON.@load params_path params seq_length
        println("✓ Parâmetros de normalização carregados")
    else
        error("Arquivo de parâmetros não encontrado: $params_path")
    end
    
    return TrainedModels(lstm_model, rnn_model, params, seq_length)
end

# Função para normalizar novos dados usando parâmetros salvos
function normalize_new_data(data::Vector{Float64}, μ::Float64, σ::Float64)
    return (data .- μ) ./ σ
end

# Função para desnormalizar predições
function denormalize_data(normalized_data::Float64, μ::Float64, σ::Float64)
    return normalized_data * σ + μ
end

# Função para preparar dados para predição
function prepare_prediction_data(stock_data, trained_models::TrainedModels, asset_name::String)
    # Encontrar parâmetros de normalização para o ativo específico
    asset_params = nothing
    for param_set in trained_models.normalization_params
        if param_set.asset == asset_name
            asset_params = param_set.params
            break
        end
    end
    
    if asset_params === nothing
        error("Parâmetros de normalização não encontrados para o ativo: $asset_name")
    end
    
    # Preparar features (mesma ordem do treinamento)
    features = hcat(
        stock_data.closing,
        stock_data.opening,
        stock_data.high,
        stock_data.low,
        log.(stock_data.volume .+ 1),
        stock_data.variation
    )
    
    # Normalizar features
    normalized_features = similar(features)
    for i in 1:size(features, 2)
        μ, σ = asset_params[i].μ, asset_params[i].σ
        normalized_features[:, i] = normalize_new_data(features[:, i], μ, σ)
    end
    
    # Usar últimos seq_length pontos para predição
    seq_length = trained_models.seq_length
    if size(normalized_features, 1) < seq_length
        error("Dados insuficientes. Necessário pelo menos $seq_length pontos.")
    end
    
    last_sequence = normalized_features[end-seq_length+1:end, :]
    
    return last_sequence, asset_params[1]  # Retorna sequência e parâmetros do preço de fechamento
end

# Função para fazer predições
function predict_next_price(stock_data, trained_models::TrainedModels, asset_name::String, model_type::Symbol = :lstm)
    # Preparar dados
    input_sequence, closing_params = prepare_prediction_data(stock_data, trained_models, asset_name)
    
    # Selecionar modelo
    model = model_type == :lstm ? trained_models.lstm_model : trained_models.rnn_model
    
    # Reset do estado do modelo
    Flux.reset!(model)
    
    # Fazer predição
    prediction_normalized = model(input_sequence')[end]
    
    # Desnormalizar predição
    prediction = denormalize_data(prediction_normalized, closing_params.μ, closing_params.σ)
    
    return prediction
end

# Função para avaliar modelos com dados históricos
function evaluate_models(stock_data, trained_models::TrainedModels, asset_name::String)
    println("Avaliando modelos para: $asset_name")
    
    if length(stock_data.closing) < trained_models.seq_length + 5
        println("Dados insuficientes para avaliação")
        return nothing
    end
    
    # Usar dados até -2 para predizer -1, e dados até -1 para predizer atual
    test_points = min(5, length(stock_data.closing) - trained_models.seq_length)
    
    lstm_errors = Float64[]
    rnn_errors = Float64[]
    
    for i in 1:test_points
        # Criar subset dos dados
        end_idx = length(stock_data.closing) - test_points + i - 1
        test_data = (
            closing = stock_data.closing[1:end_idx],
            opening = stock_data.opening[1:end_idx],
            high = stock_data.high[1:end_idx],
            low = stock_data.low[1:end_idx],
            volume = stock_data.volume[1:end_idx],
            variation = stock_data.variation[1:end_idx]
        )
        
        actual_price = stock_data.closing[end_idx + 1]
        
        # Predições
        try
            lstm_pred = predict_next_price(test_data, trained_models, asset_name, :lstm)
            rnn_pred = predict_next_price(test_data, trained_models, asset_name, :rnn)
            
            lstm_error = abs(lstm_pred - actual_price) / actual_price * 100
            rnn_error = abs(rnn_pred - actual_price) / actual_price * 100
            
            push!(lstm_errors, lstm_error)
            push!(rnn_errors, rnn_error)
            
            println("Ponto $i - Real: $(round(actual_price, digits=2)), " *
                   "LSTM: $(round(lstm_pred, digits=2)) ($(round(lstm_error, digits=2))%), " *
                   "RNN: $(round(rnn_pred, digits=2)) ($(round(rnn_error, digits=2))%)")
        catch e
            println("Erro na avaliação do ponto $i: $e")
        end
    end
    
    if !isempty(lstm_errors)
        println("\nEstatísticas de Erro:")
        println("LSTM - Erro médio: $(round(mean(lstm_errors), digits=2))%")
        println("RNN - Erro médio: $(round(mean(rnn_errors), digits=2))%")
    end
end

# Função para carregar dados de ação
function load_stock_data(filepath::String)
    data = TOML.parsefile(filepath)
    records = data["records"]
    
    return (
        closing = [record["closing"] for record in records],
        opening = [record["opening"] for record in records],
        high = [record["high"] for record in records],
        low = [record["low"] for record in records],
        volume = [record["volume"] for record in records],
        variation = [record["variation"] for record in records],
        dates = [record["date"] for record in records],
        asset_name = data["asset"]
    )
end

# Exemplo de uso
function example_usage()
    println("=== Exemplo de Uso dos Modelos Treinados ===\n")
    
    try
        # Carregar modelos treinados
        trained_models = load_trained_models()
        
        # Exemplo com CMIG4
        data_path = "../../dados/CMIG4 Dados Históricos_investing_output.toml"
        if isfile(data_path)
            stock_data = load_stock_data(data_path)
            asset_name = "CMIG4 Dados Históricos"
            
            println("Dados carregados para: $asset_name")
            println("Último preço de fechamento: $(stock_data.closing[end])")
            
            # Fazer predições
            lstm_prediction = predict_next_price(stock_data, trained_models, asset_name, :lstm)
            rnn_prediction = predict_next_price(stock_data, trained_models, asset_name, :rnn)
            
            println("\nPredições para o próximo período:")
            println("LSTM: $(round(lstm_prediction, digits=2))")
            println("RNN: $(round(rnn_prediction, digits=2))")
            
            # Avaliar modelos
            println("\n" * "="^50)
            evaluate_models(stock_data, trained_models, asset_name)
            
        else
            println("Arquivo de dados não encontrado: $data_path")
        end
        
    catch e
        println("Erro: $e")
        println("Certifique-se de que os modelos foram treinados executando o script principal primeiro.")
    end
end

# Executar exemplo
example_usage()