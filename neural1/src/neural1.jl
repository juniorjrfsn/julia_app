module neural1

    # greet() = print("Hello World!")
    # Vetor de entrada
    #X = randn(4)
    X = [0.9518042576448666, 0.7533530805209718, 0.10144084788780548, 0.952116195276538]
    
    # Pesos da primeira camada oculta
    W1 = randn(4, 4)

    println("Pesos da primeira camada oculta: ", W1)

    # Pesos da segunda camada oculta
    W2 = randn(3, 4)

    # Função de ativação sigmoide
    σ(x) = 1.0 / (1.0 + exp(-x))

    # Propagação para frente (primeira camada)
    Z1 = W1 * X
    H1 = σ.(Z1)

    # Propagação para frente (segunda camada)
    Z2 = W2 * H1
    H2 = σ.(Z2)

    println("Saída da primeira camada oculta:")
    println(H1)
    println("Saída da segunda camada oculta:")
    println(H2)

    # println("Entrada: ", input)
    # println("Saída: ", output)
end # module neural1
