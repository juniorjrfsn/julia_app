#module lstmrnntrain
#greet() = print("Hello World!")
#end # module lstmrnntrain

# projeto : lstmrnntrain
# file : lstmrnntrain/src/lstmrnntrain.jl

using Flux, TOML, Statistics, Dates, Random, BSON, StatsBase

# Configurar seed para reprodutibilidade
Random.seed!(42)

# Função para carregar dados TOML
function load_stock_data(filepath::String)
    data = TOML.parsefile(filepath)
    records = data["records"]
    
    # Extrair dados relevantes
    dates = [record["date"] for record in records]
    closing = [record["closing"] for record in records]
    volume = [record["volume"] for record in records]
    variation = [record["variation"] for record in records]
    opening = [record["opening"] for record in records]
    high = [record["high"] for record in records]
    low = [record["low"] for record in records]
    
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
end

# Função para normalizar dados
function normalize_data(data::Vector{Float64})
    μ = mean(data)
    σ = std(data)
    return (data .- μ) ./ σ, μ, σ
end

# Função para criar sequências para treinamento
function create_sequences(data::Vector{Float64}, seq_length::Int)
    X = []
    y = []
    
    for i in 1:(length(data) - seq_length)
        push!(X, data[i:(i + seq_length - 1)])
        push!(y, data[i + seq_length])
    end
    
    return hcat(X...), y
end

# Função para preparar dados multivariados
function prepare_multivariate_data(stock_data, seq_length::Int = 5)
    # Combinar features relevantes
    features = hcat(
        stock_data.closing,
        stock_data.opening,
        stock_data.high,
        stock_data.low,
        log.(stock_data.volume .+ 1),  # Log do volume para reduzir escala
        stock_data.variation
    )
    
    # Normalizar cada feature
    normalized_features = similar(features)
    normalization_params = []
    
    for i in 1:size(features, 2)
        normalized_features[:, i], μ, σ = normalize_data(features[:, i])
        push!(normalization_params, (μ = μ, σ = σ))
    end
    
    # Criar sequências
    X = []
    y = []
    
    for i in 1:(size(normalized_features, 1) - seq_length)
        push!(X, normalized_features[i:(i + seq_length - 1), :])
        push!(y, normalized_features[i + seq_length, 1])  # Prever preço de fechamento
    end
    
    # Converter para arrays 3D (samples, timesteps, features)
    X_tensor = cat(X..., dims = 3)
    X_tensor = permutedims(X_tensor, (3, 1, 2))  # (samples, timesteps, features)
    
    return X_tensor, y, normalization_params
end

# Definir modelo LSTM
function create_lstm_model(input_size::Int, hidden_size::Int = 32, output_size::Int = 1)
    return Chain(
        LSTM(input_size => hidden_size),
        Dense(hidden_size => hidden_size, relu),
        Dropout(0.2),
        Dense(hidden_size => output_size)
    )
end

# Definir modelo RNN simples
function create_rnn_model(input_size::Int, hidden_size::Int = 32, output_size::Int = 1)
    return Chain(
        RNN(input_size => hidden_size, tanh),
        Dense(hidden_size => hidden_size, relu),
        Dropout(0.2),
        Dense(hidden_size => output_size)
    )
end

# Função de treinamento (VERSÃO CORRIGIDA)
function train_model(model, X, y, epochs::Int = 100, lr::Float64 = 0.001)
    # Preparar dados para treinamento
    X_train = [Float32.(X[i, :, :]) for i in 1:size(X, 1)]  # Converter para Float32
    y_train = Float32.(collect(y))  # Garantir que y seja um vetor de Float32
    
    # Configurar otimizador usando a nova API
    opt_state = Flux.setup(Adam(lr), model)
    
    # Histórico de perdas
    losses = Float64[]
    
    println("Iniciando treinamento...")
    
    for epoch in 1:epochs
        total_loss = 0.0
        
        for i in 1:length(X_train)
            # Reset estado para cada sequência
            Flux.reset!(model)
            
            # Calcular gradientes e atualizar parâmetros
            loss_val, grads = Flux.withgradient(model) do m
                pred = m(X_train[i]')
                # pred é uma matriz, pegamos o último valor da última coluna
                prediction = pred[:, end][1]  # Primeiro elemento da última coluna
                return Flux.mse(prediction, y_train[i])
            end
            
            # Verificar se o gradiente é válido
            if !isnothing(grads[1])
                Flux.update!(opt_state, model, grads[1])
            end
            
            total_loss += loss_val
        end
        
        avg_loss = total_loss / length(X_train)
        push!(losses, avg_loss)
        
        if epoch % 20 == 0
            println("Epoch $epoch: Loss = $(round(avg_loss, digits=6))")
        end
    end
    
    return model, losses
end

# Função para fazer predições
function predict_sequence(model, X_input)
    Flux.reset!(model)
    pred = model(X_input')
    return pred[:, end][1]  # Retornar a predição final
end

# Função principal
function main()
    println("=== Treinamento de Modelos LSTM e RNN para Dados de Ações ===\n")
    
    # Definir caminhos dos arquivos
    data_dir = "../../../dados/ativos"
    files = [
        "CMIG4 Dados Históricos_investing_output.toml",
        "CPFE3 Dados Históricos_investing_output.toml",
        "DIRR3 Dados Históricos_investing_output.toml",
        "EGIE3 Dados Históricos_investing_output.toml"
    ]
    
    # Carregar e processar dados de todas as ações
    all_data = []
    
    for file in files
        filepath = joinpath(data_dir, file)
        if isfile(filepath)
            println("Carregando dados de: $file")
            stock_data = load_stock_data(filepath)
            push!(all_data, stock_data)
        else
            println("Arquivo não encontrado: $filepath")
        end
    end
    
    if isempty(all_data)
        println("Nenhum dado foi carregado. Verifique os caminhos dos arquivos.")
        return
    end
    
    # Combinar dados de todas as ações para treinamento conjunto
    println("\nPreparando dados para treinamento...")
    
    all_X = []
    all_y = []
    all_params = []
    
    seq_length = 5  # Usar últimos 5 dias para prever o próximo
    
    for (i, stock_data) in enumerate(all_data)
        X, y, norm_params = prepare_multivariate_data(stock_data, seq_length)
        
        if size(X, 1) > 0  # Verificar se há dados suficientes
            push!(all_X, X)
            push!(all_y, y)
            push!(all_params, (asset = stock_data.asset_name, params = norm_params))
            
            println("$(stock_data.asset_name): $(size(X, 1)) sequências criadas")
        end
    end
    
    # Combinar todos os dados
    if !isempty(all_X)
        X_combined = vcat(all_X...)
        y_combined = vcat(all_y...)
        
        println("\nDados combinados: $(size(X_combined, 1)) sequências, $(size(X_combined, 3)) features")
        
        # Treinar modelo LSTM
        println("\n=== Treinando Modelo LSTM ===")
        lstm_model = create_lstm_model(size(X_combined, 3), 32, 1)
        lstm_trained, lstm_losses = train_model(lstm_model, X_combined, y_combined, 100, 0.001)
        
        # Treinar modelo RNN
        println("\n=== Treinando Modelo RNN ===")
        rnn_model = create_rnn_model(size(X_combined, 3), 32, 1)
        rnn_trained, rnn_losses = train_model(rnn_model, X_combined, y_combined, 100, 0.001)
        
        # Salvar modelos e parâmetros
        println("\n=== Salvando Modelos Treinados ===")
        
        # Criar diretório para salvar modelos
        save_dir = "../../../dados/modelos_treinados"
        mkpath(save_dir)
        
        # Salvar modelo LSTM
        lstm_save_path = joinpath(save_dir, "lstm_model.bson")
        BSON.@save lstm_save_path model = lstm_trained
        println("Modelo LSTM salvo em: $lstm_save_path")
        
        # Salvar modelo RNN
        rnn_save_path = joinpath(save_dir, "rnn_model.bson")
        BSON.@save rnn_save_path model = rnn_trained
        println("Modelo RNN salvo em: $rnn_save_path")
        
        # Salvar parâmetros de normalização
        params_save_path = joinpath(save_dir, "normalization_params.bson")
        BSON.@save params_save_path params = all_params seq_length = seq_length
        println("Parâmetros de normalização salvos em: $params_save_path")
        
        # Salvar histórico de perdas
        losses_save_path = joinpath(save_dir, "training_losses.bson")
        BSON.@save losses_save_path lstm_losses = lstm_losses rnn_losses = rnn_losses
        println("Histórico de perdas salvo em: $losses_save_path")
        
        # Teste rápido dos modelos
        println("\n=== Teste Rápido dos Modelos ===")
        test_sample = Float32.(X_combined[1, :, :])
        
        try
            lstm_pred = predict_sequence(lstm_trained, test_sample)
            rnn_pred = predict_sequence(rnn_trained, test_sample)
            
            println("Predição LSTM: $(round(lstm_pred, digits=4))")
            println("Predição RNN: $(round(rnn_pred, digits=4))")
            println("Valor real: $(round(y_combined[1], digits=4))")
        catch e
            println("Erro no teste: $e")
        end
        
        # Relatório final
        println("\n=== Relatório de Treinamento ===")
        println("Perda final LSTM: $(round(lstm_losses[end], digits=6))")
        println("Perda final RNN: $(round(rnn_losses[end], digits=6))")
        println("Sequências de treinamento: $(length(y_combined))")
        println("Features por sequência: $(size(X_combined, 3))")
        println("Comprimento da sequência: $seq_length")
        
        println("\n=== Arquivos Salvos ===")
        for file in readdir(save_dir, join=true)
            println("- $file")
        end
        
        println("\nTreinamento concluído com sucesso!")
        
        # Exemplo de como carregar os modelos posteriormente
        println("\n=== Exemplo de Carregamento ===")
        println("Para carregar os modelos posteriormente, use:")
        println("using BSON")
        println("BSON.@load \"$lstm_save_path\" model")
        println("lstm_model = model")
        
    else
        println("Não foi possível criar sequências de treinamento.")
    end
end

# Executar treinamento
main()





#  julia lstmrnntrain.jl