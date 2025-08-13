#module lstmrnntrain
#greet() = print("Hello World!")
#end # module lstmrnntrain

# projeto : lstmrnntrain
# file : lstmrnntrain/src/lstmrnntrain.jl


using Flux, TOML, Statistics, Dates, Random, JSON, StatsBase

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

# Função de treinamento
function train_model(model, X, y, epochs::Int = 100, lr::Float64 = 0.001)
    # Preparar dados para treinamento
    X_train = [Float32.(X[i, :, :]) for i in 1:size(X, 1)]  # Converter para Float32
    y_train = Float32.(collect(y))  # Garantir que y seja um vetor de Float32
    
    # Configurar otimizador
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

# Função para extrair parâmetros do modelo para serialização (CORRIGIDA)
function extract_model_params(model)
    params = []
    for layer in model.layers
        if layer isa Dense
            push!(params, Dict(
                "type" => "Dense",
                "weight" => Array(layer.weight),
                "bias" => Array(layer.bias),
                "activation" => string(layer.σ)
            ))
        elseif layer isa LSTM
            # Corrigir acesso aos parâmetros do LSTM
            cell = layer.cell
            Wi = cell.Wi  # Matriz de pesos de entrada
            Wh = cell.Wh  # Matriz de pesos ocultos
            input_size = size(Wi, 2)  # Segunda dimensão de Wi é input_size
            hidden_size = size(Wh, 2)  # Segunda dimensão de Wh é hidden_size
            
            # Verificar se existe bias - alguns LSTMs podem não ter
            bias_data = nothing
            try
                if hasfield(typeof(cell), :bias)
                    bias_data = Array(cell.bias)
                elseif hasfield(typeof(cell), :b)
                    bias_data = Array(cell.b)
                else
                    # Procurar por outros possíveis nomes de bias
                    for fieldname in fieldnames(typeof(cell))
                        if String(fieldname) in ["bias", "b", "bi", "bh"]
                            bias_data = Array(getfield(cell, fieldname))
                            break
                        end
                    end
                end
            catch e
                println("Aviso: Não foi possível extrair bias do LSTM: $e")
                bias_data = nothing
            end
            
            lstm_dict = Dict(
                "type" => "LSTM",
                "input_size" => input_size,
                "hidden_size" => hidden_size,
                "Wi" => Array(Wi),
                "Wh" => Array(Wh)
            )
            
            if bias_data !== nothing
                lstm_dict["bias"] = bias_data
            end
            
            push!(params, lstm_dict)
            
        elseif layer isa RNN
            # Corrigir acesso aos parâmetros do RNN
            cell = layer.cell
            Wi = cell.Wi
            Wh = cell.Wh
            input_size = size(Wi, 2)
            hidden_size = size(Wh, 2)
            
            # Verificar se existe bias
            bias_data = nothing
            try
                if hasfield(typeof(cell), :bias)
                    bias_data = Array(cell.bias)
                elseif hasfield(typeof(cell), :b)
                    bias_data = Array(cell.b)
                else
                    # Procurar por outros possíveis nomes de bias
                    for fieldname in fieldnames(typeof(cell))
                        if String(fieldname) in ["bias", "b", "bi", "bh"]
                            bias_data = Array(getfield(cell, fieldname))
                            break
                        end
                    end
                end
            catch e
                println("Aviso: Não foi possível extrair bias do RNN: $e")
                bias_data = nothing
            end
            
            rnn_dict = Dict(
                "type" => "RNN",
                "input_size" => input_size,
                "hidden_size" => hidden_size,
                "Wi" => Array(Wi),
                "Wh" => Array(Wh),
                "activation" => string(cell.σ)
            )
            
            if bias_data !== nothing
                rnn_dict["bias"] = bias_data
            end
            
            push!(params, rnn_dict)
            
        elseif layer isa Dropout
            push!(params, Dict(
                "type" => "Dropout",
                "p" => layer.p
            ))
        end
    end
    return params
end

# Função para descobrir todos os arquivos TOML na pasta
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
    
    return sort(toml_files)  # Ordenar alfabeticamente
end

# Função principal
function main()
    println("=== Treinamento de Modelos LSTM e RNN para Dados de Ações ===\n")
    
    # Definir caminho do diretório de dados
    data_dir = "../../../dados/ativos"
    
    # Descobrir todos os arquivos TOML na pasta
    println("Procurando arquivos TOML em: $data_dir")
    files = find_toml_files(data_dir)
    
    if isempty(files)
        println("Nenhum arquivo TOML encontrado em: $data_dir")
        return
    end
    
    println("Arquivos TOML encontrados:")
    for (i, file) in enumerate(files)
        println("  $i. $file")
    end
    println()
    
    # Carregar e processar dados de todas as ações
    all_data = []
    successful_loads = 0
    failed_loads = 0
    
    for file in files
        filepath = joinpath(data_dir, file)
        try
            println("Carregando dados de: $file")
            stock_data = load_stock_data(filepath)
            push!(all_data, stock_data)
            successful_loads += 1
        catch e
            println("  ❌ Erro ao carregar $file: $e")
            failed_loads += 1
        end
    end
    
    println("\nResumo do carregamento:")
    println("  ✅ Arquivos carregados com sucesso: $successful_loads")
    println("  ❌ Arquivos com erro: $failed_loads")
    println("  📁 Total de arquivos processados: $(length(files))")
    
    if isempty(all_data)
        println("\nNenhum dado foi carregado. Verifique os arquivos TOML.")
        return
    end
    
    # Combinar dados de todas as ações para treinamento conjunto
    println("\nPreparando dados para treinamento...")
    
    all_X = []
    all_y = []
    all_params = []
    
    seq_length = 5  # Usar últimos 5 dias para prever o próximo
    
    for (i, stock_data) in enumerate(all_data)
        try
            X, y, norm_params = prepare_multivariate_data(stock_data, seq_length)
            
            if size(X, 1) > 0  # Verificar se há dados suficientes
                push!(all_X, X)
                push!(all_y, y)
                push!(all_params, Dict(
                    "asset" => stock_data.asset_name,
                    "params" => [Dict("mu" => p.μ, "sigma" => p.σ) for p in norm_params]
                ))
                
                println("  ✅ $(stock_data.asset_name): $(size(X, 1)) sequências criadas")
            else
                println("  ⚠️  $(stock_data.asset_name): Dados insuficientes para criar sequências")
            end
        catch e
            println("  ❌ Erro ao processar $(stock_data.asset_name): $e")
        end
    end
    
    # Combinar todos os dados
    if !isempty(all_X)
        X_combined = vcat(all_X...)
        y_combined = vcat(all_y...)
        
        println("\n📊 Dados combinados:")
        println("  • Sequências de treinamento: $(size(X_combined, 1))")
        println("  • Features por sequência: $(size(X_combined, 3))")
        println("  • Comprimento da sequência: $seq_length")
        
        # Treinar modelo LSTM
        println("\n=== Treinando Modelo LSTM ===")
        lstm_model = create_lstm_model(size(X_combined, 3), 32, 1)
        lstm_trained, lstm_losses = train_model(lstm_model, X_combined, y_combined, 100, 0.001)
        
        # Treinar modelo RNN
        println("\n=== Treinando Modelo RNN ===")
        rnn_model = create_rnn_model(size(X_combined, 3), 32, 1)
        rnn_trained, rnn_losses = train_model(rnn_model, X_combined, y_combined, 100, 0.001)
        
        # Salvar modelos e parâmetros em JSON
        println("\n=== Salvando Modelos Treinados em JSON ===")
        
        # Criar diretório para salvar modelos
        save_dir = "../../../dados/modelos_treinados"
        mkpath(save_dir)
        
        # Extrair parâmetros dos modelos
        try
            println("Extraindo parâmetros do modelo LSTM...")
            lstm_params = extract_model_params(lstm_trained)
            
            println("Extraindo parâmetros do modelo RNN...")
            rnn_params = extract_model_params(rnn_trained)
            
            # Preparar dados para salvar
            training_data = Dict(
                "training_info" => Dict(
                    "timestamp" => string(now()),
                    "total_sequences" => length(y_combined),
                    "sequence_length" => seq_length,
                    "features_count" => size(X_combined, 3),
                    "epochs" => 100,
                    "learning_rate" => 0.001,
                    "assets_trained" => [p["asset"] for p in all_params]
                ),
                "lstm_model" => Dict(
                    "architecture" => lstm_params,
                    "final_loss" => lstm_losses[end],
                    "training_losses" => lstm_losses
                ),
                "rnn_model" => Dict(
                    "architecture" => rnn_params,
                    "final_loss" => rnn_losses[end],
                    "training_losses" => rnn_losses
                ),
                "normalization_params" => all_params,
                "feature_names" => [
                    "closing", "opening", "high", "low", "log_volume", "variation"
                ]
            )
            
            # Salvar em JSON
            json_save_path = joinpath(save_dir, "trained_models.json")
            open(json_save_path, "w") do file
                JSON.print(file, training_data, 2)  # Indentação de 2 espaços
            end
            println("  ✅ Modelos salvos em JSON: $json_save_path")
            
        catch e
            println("  ❌ Erro ao extrair/salvar parâmetros: $e")
            println("  ℹ️  Continuando sem salvar os modelos...")
        end
        
        # Salvar também um arquivo de configuração separado
        config_data = Dict(
            "model_config" => Dict(
                "sequence_length" => seq_length,
                "features_count" => size(X_combined, 3),
                "hidden_size" => 32,
                "feature_names" => [
                    "closing", "opening", "high", "low", "log_volume", "variation"
                ]
            ),
            "usage_instructions" => Dict(
                "input_format" => "Array of shape [sequence_length, features_count]",
                "output_format" => "Single predicted closing price (normalized)",
                "preprocessing" => "Apply normalization using provided parameters",
                "postprocessing" => "Denormalize using closing price parameters"
            )
        )
        
        config_save_path = joinpath(save_dir, "model_config.json")
        open(config_save_path, "w") do file
            JSON.print(file, config_data, 2)
        end
        println("  ✅ Configuração salva em: $config_save_path")
        
        # Teste rápido dos modelos
        println("\n=== Teste Rápido dos Modelos ===")
        test_sample = Float32.(X_combined[1, :, :])
        
        try
            lstm_pred = predict_sequence(lstm_trained, test_sample)
            rnn_pred = predict_sequence(rnn_trained, test_sample)
            
            println("  • Predição LSTM: $(round(lstm_pred, digits=4))")
            println("  • Predição RNN: $(round(rnn_pred, digits=4))")
            println("  • Valor real: $(round(y_combined[1], digits=4))")
        catch e
            println("  ❌ Erro no teste: $e")
        end
        
        # Relatório final
        println("\n=== Relatório Final de Treinamento ===")
        println("📈 Desempenho dos Modelos:")
        println("  • Perda final LSTM: $(round(lstm_losses[end], digits=6))")
        println("  • Perda final RNN: $(round(rnn_losses[end], digits=6))")
        println("  • Melhoria LSTM: $(round((lstm_losses[1] - lstm_losses[end])/lstm_losses[1]*100, digits=2))%")
        println("  • Melhoria RNN: $(round((rnn_losses[1] - rnn_losses[end])/rnn_losses[1]*100, digits=2))%")
        
        println("\n📊 Dados de Treinamento:")
        println("  • Arquivos TOML processados: $(length(files))")
        println("  • Ativos com dados válidos: $(length(all_params))")
        println("  • Sequências de treinamento: $(length(y_combined))")
        println("  • Features por sequência: $(size(X_combined, 3))")
        println("  • Comprimento da sequência: $seq_length")
        
        println("\n📝 Arquivos Gerados:")
        for file in readdir(save_dir, join=true)
            if endswith(file, ".json")
                println("  • $file")
            end
        end
        
        println("\n✅ Treinamento concluído com sucesso!")
        
        # Exemplo de uso em outras linguagens
        println("\n=== Exemplo de Uso em Outras Linguagens ===")
        println("Os modelos foram salvos em formato JSON e podem ser carregados em qualquer linguagem:")
        println("  • Python: json.load()")
        println("  • JavaScript: JSON.parse()")
        println("  • Java: Jackson ou Gson")
        println("  • C#: JsonSerializer")
        println("  • R: jsonlite::fromJSON()")
        
        println("\nPara usar os modelos, carregue o arquivo 'trained_models.json' e")
        println("implemente as camadas LSTM/RNN usando os parâmetros salvos.")
        
    else
        println("\n❌ Não foi possível criar sequências de treinamento.")
        println("Verifique se os arquivos TOML contêm dados válidos.")
    end
end

# Executar treinamento
main()



# Para executar: 
#  julia lstmrnntrain.jl