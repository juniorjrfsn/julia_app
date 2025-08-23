# projeto : z_hrm
# file : z_hrm/src/z_hrm.jl
module z_hrm 
    using Random
    using LinearAlgebra
    using Statistics

    println("=== Iniciando HRM Neural Network ===\n")

    # Estrutura para a Rede Neural HRM
    mutable struct HRMNeuralNetwork
        layers::Vector{Int}           
        weights::Vector{Matrix{Float64}}  
        biases::Vector{Vector{Float64}}   
        learning_rate::Float64        
        regularization::Float64       
        hierarchy_levels::Int         
    end

    # Construtor da rede neural HRM
    function HRMNeuralNetwork(layers::Vector{Int}; 
                             learning_rate::Float64 = 0.01, 
                             regularization::Float64 = 0.001,
                             hierarchy_levels::Int = 3)
        
        weights = Vector{Matrix{Float64}}()
        biases = Vector{Vector{Float64}}()
        
        # Inicialização Xavier/Glorot
        for i in 1:length(layers)-1
            n_in, n_out = layers[i], layers[i+1]
            w = randn(n_out, n_in) * sqrt(2.0 / (n_in + n_out))
            b = zeros(n_out)
            push!(weights, w)
            push!(biases, b)
        end
        
        return HRMNeuralNetwork(layers, weights, biases, learning_rate, 
                               regularization, hierarchy_levels)
    end

    # Função de ativação sigmoid
    function sigmoid(x::Float64)
        return 1.0 / (1.0 + exp(-clamp(x, -500, 500)))
    end

    # Derivada da sigmoid
    function sigmoid_derivative(x::Float64)
        s = sigmoid(x)
        return s * (1.0 - s)
    end

    # Função ReLU
    function relu(x::Float64)
        return max(0.0, x)
    end

    # Derivada da ReLU
    function relu_derivative(x::Float64)
        return x > 0 ? 1.0 : 0.0
    end

    # Forward pass
    function forward_pass(network::HRMNeuralNetwork, input::Vector{Float64})
        activations = [input]
        z_values = Vector{Vector{Float64}}()
        
        current_input = input
        
        for i in 1:length(network.weights)
            z = network.weights[i] * current_input .+ network.biases[i]
            push!(z_values, z)
            
            if i < length(network.weights)
                current_input = [relu(zi) for zi in z]
            else
                current_input = [sigmoid(zi) for zi in z]
            end
            
            push!(activations, current_input)
        end
        
        return activations, z_values
    end

    # Cálculo do risco hierárquico
    function hierarchical_risk(network::HRMNeuralNetwork, 
                             predictions::Vector{Float64}, 
                             targets::Vector{Float64})
        
        base_error = mean((predictions .- targets).^2)
        complexity_penalty = 0.0
        
        for (level, weights) in enumerate(network.weights)
            level_penalty = sum(weights.^2) * (level / network.hierarchy_levels)
            complexity_penalty += level_penalty
        end
        
        total_risk = base_error + network.regularization * complexity_penalty
        return total_risk
    end

    # Backward pass com HRM
    function backward_pass(network::HRMNeuralNetwork, 
                          activations::Vector{Vector{Float64}},
                          z_values::Vector{Vector{Float64}},
                          targets::Vector{Float64})
        
        n_layers = length(network.weights)
        weight_gradients = Vector{Matrix{Float64}}(undef, n_layers)
        bias_gradients = Vector{Vector{Float64}}(undef, n_layers)
        
        output_error = activations[end] .- targets
        delta = output_error .* [sigmoid_derivative(z) for z in z_values[end]]
        
        weight_gradients[end] = delta * activations[end-1]'
        bias_gradients[end] = delta
        
        for i in (n_layers-1):-1:1
            delta = (network.weights[i+1]' * delta) .* [relu_derivative(z) for z in z_values[i]]
            
            hierarchy_factor = (i / network.hierarchy_levels)
            regularization_term = 2 * network.regularization * hierarchy_factor * network.weights[i]
            
            weight_gradients[i] = delta * activations[i]' .+ regularization_term
            bias_gradients[i] = delta
        end
        
        return weight_gradients, bias_gradients
    end

    # Atualização dos pesos
    function update_weights!(network::HRMNeuralNetwork,
                            weight_gradients::Vector{Matrix{Float64}},
                            bias_gradients::Vector{Vector{Float64}})
        
        for i in 1:length(network.weights)
            network.weights[i] .-= network.learning_rate .* weight_gradients[i]
            network.biases[i] .-= network.learning_rate .* bias_gradients[i]
        end
    end

    # Treinamento com HRM
    function train_hrm!(network::HRMNeuralNetwork, 
                       X_train::Matrix{Float64}, 
                       y_train::Vector{Float64},
                       epochs::Int = 1000,
                       batch_size::Int = 32,
                       verbose::Bool = true)
        
        n_samples = size(X_train, 2)
        losses = Float64[]
        
        for epoch in 1:epochs
            indices = shuffle(1:n_samples)
            epoch_loss = 0.0
            
            for i in 1:batch_size:n_samples
                batch_end = min(i + batch_size - 1, n_samples)
                batch_indices = indices[i:batch_end]
                
                batch_loss = 0.0
                total_weight_grad = [zeros(size(w)) for w in network.weights]
                total_bias_grad = [zeros(size(b)) for b in network.biases]
                
                for idx in batch_indices
                    x_sample = X_train[:, idx]
                    y_sample = [y_train[idx]]
                    
                    activations, z_values = forward_pass(network, x_sample)
                    risk = hierarchical_risk(network, activations[end], y_sample)
                    batch_loss += risk
                    
                    w_grad, b_grad = backward_pass(network, activations, z_values, y_sample)
                    
                    for j in 1:length(total_weight_grad)
                        total_weight_grad[j] .+= w_grad[j]
                        total_bias_grad[j] .+= b_grad[j]
                    end
                end
                
                batch_length = length(batch_indices)
                for j in 1:length(total_weight_grad)
                    total_weight_grad[j] ./= batch_length
                    total_bias_grad[j] ./= batch_length
                end
                
                update_weights!(network, total_weight_grad, total_bias_grad)
                epoch_loss += batch_loss / batch_length
            end
            
            avg_loss = epoch_loss / ceil(n_samples / batch_size)
            push!(losses, avg_loss)
            
            if verbose && epoch % 100 == 0
                println("Época $epoch: Loss = $(round(avg_loss, digits=6))")
            end
        end
        
        return losses
    end

    # Predição
    function predict(network::HRMNeuralNetwork, X::Matrix{Float64})
        predictions = Float64[]
        
        for i in 1:size(X, 2)
            activations, _ = forward_pass(network, X[:, i])
            push!(predictions, activations[end][1])
        end
        
        return predictions
    end

    # EXECUTAR EXEMPLO
    println("=== Exemplo de Rede Neural com HRM ===\n")

    # Gerar dados de exemplo
    Random.seed!(42)
    n_samples = 1000

    X = reshape(collect(range(-2, 2, length=n_samples)), 1, :)
    y = vec(0.5 .* X[1,:].^2 .+ 0.3 .* X[1,:] .+ 0.1 .+ 0.1 .* randn(n_samples))

    # Dividir em treino e teste
    train_size = Int(0.8 * n_samples)
    X_train = X[:, 1:train_size]
    y_train = y[1:train_size]
    X_test = X[:, train_size+1:end]
    y_test = y[train_size+1:end]

    # Criar e treinar a rede
    network = HRMNeuralNetwork([1, 20, 15, 1], 
                              learning_rate=0.001, 
                              regularization=0.01,
                              hierarchy_levels=3)

    println("Treinando rede neural com HRM...")
    # CORREÇÃO: usar argumentos posicionais ao invés de nomeados
    losses = train_hrm!(network, X_train, y_train, 500, 16, true)

    # Fazer predições
    train_pred = predict(network, X_train)
    test_pred = predict(network, X_test)

    # Calcular métricas
    train_mse = mean((train_pred .- y_train).^2)
    test_mse = mean((test_pred .- y_test).^2)

    println("\n=== Resultados ===")
    println("MSE Treino: $(round(train_mse, digits=6))")
    println("MSE Teste: $(round(test_mse, digits=6))")

    # Mostrar algumas predições
    println("\n=== Primeiras 10 Predições vs Reais (Teste) ===")
    for i in 1:min(10, length(test_pred))
        println("Pred: $(round(test_pred[i], digits=4)) | Real: $(round(y_test[i], digits=4)) | Erro: $(round(abs(test_pred[i] - y_test[i]), digits=4))")
    end

    println("\n✓ Script HRM executado com sucesso!")
    println("Loss final: $(round(losses[end], digits=6))")
    println("Número de epochs: $(length(losses))")

    # Verificar convergência
    if length(losses) > 50
        convergencia = abs(losses[end] - losses[end-50]) / losses[end-50]
        println("Taxa de convergência (últimas 50 epochs): $(round(convergencia * 100, digits=2))%")
    end

    # Exportar funções principais
    export HRMNeuralNetwork, train_hrm!, predict
    export sigmoid, relu, forward_pass, hierarchical_risk, backward_pass

end # module z_hrm