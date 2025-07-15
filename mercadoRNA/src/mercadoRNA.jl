module mercadoRNA

using CSV, DataFrames, Flux, Statistics, BSON

# Função de saudação
greet() = println("Olá! Bem-vindo ao sistema de previsão de preços de ações.")

# Estrutura para armazenar o modelo e parâmetros de normalização
struct ModeloPrevisao
    model
    μ::Float32
    σ::Float32
end

# Carrega e preprocessa os dados
function load_data()
    try
        data = CSV.read("dados/WEGE3.csv", DataFrame; stringtype=String)
        
        # Função para limpar e converter preços
        clean_price(value) = parse(Float32, replace(string(value), "," => "."))
        
        # Extrai e limpa a coluna de preços (Último)
        prices = Float32[clean_price(val) for val in data.Último]
        
        # Inverte para ordem cronológica correta (mais antigo primeiro)
        return reverse(prices)
    catch e
        println("Erro ao carregar dados: ", e)
        return Float32[]
    end
end

# Normaliza os dados
function normalize(data)
    μ = mean(data)
    σ = std(data)
    normalized_data = σ ≈ 0 ? (data .- μ) : (data .- μ) ./ σ
    return normalized_data, μ, σ
end

# Cria sequências para treinamento
function create_sequences(data, seq_length)
    xs = Vector{Vector{Float32}}[]  # Array para armazenar sequências
    ys = Float32[]                  # Array para armazenar valores alvo
    for i in 1:(length(data) - seq_length)
        push!(xs, data[i:i+seq_length-1])
        push!(ys, data[i+seq_length])
    end
    return hcat(xs...), reshape(ys, 1, :)
end

# Define a arquitetura do modelo
function build_model(input_dim, hidden_dim, output_dim)
    return Chain(
        LSTM(input_dim => hidden_dim),
        Dense(hidden_dim => output_dim)
    )
end

# Função para treinar e salvar o modelo
function train_model(;seq_length=10, hidden_dim=50, epochs=100, lr=0.01)
    println("\n=== INICIANDO TREINAMENTO ===")
    
    # 1. Carregar e preparar dados
    prices = load_data()
    if isempty(prices)
        error("Dados não carregados corretamente")
    end
    
    normalized_data, μ, σ = normalize(prices)
    X, y = create_sequences(normalized_data, seq_length)
    
    # 2. Dividir dados (80% treino, 20% teste)
    split_idx = Int(floor(0.8 * size(X, 2)))
    X_train, y_train = X[:, 1:split_idx], y[:, 1:split_idx]
    X_test, y_test = X[:, split_idx+1:end], y[:, split_idx+1:end]
    
    # 3. Reshape para formato LSTM (seq_len, batch, n_samples)
    X_train = reshape(X_train, seq_length, 1, size(X_train, 2))
    X_test = reshape(X_test, seq_length, 1, size(X_test, 2))
    
    # 4. Construir e treinar modelo
    model = build_model(seq_length, hidden_dim, 1)
    opt = Flux.ADAM(lr)
    
    # Função de perda
    loss(x, y) = Flux.mse(model(x), y)
    
    println("Iniciando treinamento com $(size(X_train, 3)) amostras...")
    
    # Treinamento
    for epoch in 1:epochs
        Flux.reset!(model)
        Flux.train!(loss, Flux.params(model), [(X_train, y_train)], opt)
        
        if epoch % 10 == 0 || epoch == 1 || epoch == epochs
            train_loss = loss(X_train, y_train)
            test_loss = loss(X_test, y_test)
            println("Época $epoch/$epochs - Perda Treino: $(round(train_loss, digits=5)) | Teste: $(round(test_loss, digits=5))")
        end
    end
    
    # 5. Salvar modelo
    modelo = ModeloPrevisao(model, μ, σ)
    BSON.@save "modelo_treinado.bson" modelo
    
    println("\nModelo treinado e salvo com sucesso!")
    return modelo
end

# Carrega modelo salvo
function load_saved_model()
    try
        BSON.@load "modelo_treinado.bson" modelo
        return modelo
    catch e
        println("Modelo não encontrado ou erro ao carregar: ", e)
        return nothing
    end
end

# Faz previsão com o modelo treinado
function predict()
    println("\n=== INICIANDO PREVISÃO ===")
    
    # Tentar carregar modelo salvo
    modelo = load_saved_model()
    
    # Se não existir, treinar novo
    if isnothing(modelo)
        println("Modelo não encontrado. Treinando novo modelo...")
        modelo = train_model()
    end
    
    # Carregar dados mais recentes
    prices = load_data()
    if isempty(prices)
        error("Dados não carregados corretamente")
    end
    
    # Preparar última sequência
    seq_length = length(first(modelo.model.layers[1].state).hidden)
    normalized_prices = (prices .- modelo.μ) ./ modelo.σ
    last_sequence = normalized_prices[end-seq_length+1:end]
    input_data = reshape(last_sequence, seq_length, 1, 1)
    
    # Fazer previsão
    Flux.reset!(modelo.model)
    pred_normalized = modelo.model(input_data)[1]
    prediction = pred_normalized * modelo.σ + modelo.μ
    
    # Resultado
    last_price = prices[end]
    change = (prediction - last_price) / last_price * 100
    
    println("\nÚltimo preço conhecido: R\$ $(round(last_price, digits=2))")
    println("Previsão próximo preço: R\$ $(round(prediction, digits=2))")
    println("Variação estimada: $(round(change, digits=2))%")
    println("\n=== PREVISÃO CONCLUÍDA ===")
    
    return prediction
end

# Função para treinar explicitamente
function train()
    modelo = train_model()
    return modelo
end

end # module

# Interface de linha de comando
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) > 0
        command = ARGS[1]
        if command == "treino"
            println("Iniciando treinamento...")
            mercadoRNA.train()
        elseif command == "prever"
            println("Preparando previsão...")
            mercadoRNA.predict()
        else
            println("Comando inválido. Use 'treino' ou 'prever'")
            mercadoRNA.greet()
        end
    else
        println("Nenhum comando fornecido. Use 'treino' ou 'prever'")
        mercadoRNA.greet()
    end
end




## Execute ##
# $ cd .\mercadoRNA\src\
# $ julia perceptronxor_4.jl
# $ mercadoRNA.jl

#  
# $  julia mercadoRNA.jl treino 
# $  julia mercadoRNA.jl prever  
