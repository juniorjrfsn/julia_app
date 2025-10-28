# Porta lógica XOR usando uma rede neural simples em Julia com Flux.jl
# file: xor.jl

 
 # xor.jl
using Random

# Funções de ativação
sigmoid(x) = 1.0 ./ (1.0 .+ exp.(-x))
dsigmoid(x) = x .* (1 .- x)  # derivada em termos da saída da sigmoide

# Dados da tabela verdade XOR
X = Float32[
    0 0;
    0 1;
    1 0;
    1 1
]

y = Float32[
    0;
    1;
    1;
    0
]

# Estrutura da rede: 2 → 2 → 1
input_size = 2
hidden_size = 4
output_size = 1
lr = 0.1   # taxa de aprendizado
epochs = 10000

# Inicialização aleatória dos pesos
Random.seed!(1234) 
W1 = rand(Float32, input_size, hidden_size) .* 0.7
b1 = rand(Float32, 1, hidden_size) .* 0.7
W2 = rand(Float32, hidden_size, output_size) .* 0.7
b2 = rand(Float32, 1, output_size) .* 0.7

println("Treinando a rede...")

for epoch in 1:epochs
    # Forward pass
    z1 = X * W1 .+ b1
    a1 = sigmoid.(z1)
    z2 = a1 * W2 .+ b2
    a2 = sigmoid.(z2)

    # Cálculo do erro
    loss = sum((y .- a2).^2) / size(y,1)

    # Backpropagation
    error_output = (y .- a2) .* dsigmoid.(a2)
    error_hidden = (error_output * W2') .* dsigmoid.(a1)

    # Atualização dos pesos
    W2 .+= (a1' * error_output) .* lr
    b2 .+= sum(error_output, dims=1) .* lr
    W1 .+= (X' * error_hidden) .* lr
    b1 .+= sum(error_hidden, dims=1) .* lr

    # Mostrar perda a cada 1000 épocas
    if epoch % 1000 == 0
        println("Epoch $epoch - Loss: $loss")
    end
end

# Testando resultados finais
println("\nResultados da porta lógica XOR:")
for i in 1:size(X,1)
    z1 = X[i,:]' * W1 .+ b1
    a1 = sigmoid.(z1)
    z2 = a1 * W2 .+ b2
    a2 = sigmoid.(z2)
    println("Entrada: $(X[i,:]) -> Saída: $(round(a2[1]; digits=3))")
end

# Modo interativo
println("\n--- Modo Interativo ---")
while true
    println("Digite os valores de entrada (0 ou 1) para a porta lógica XOR (ou Ctrl+C para sair):")
    x1 = parse(Float32, readline())
    x2 = parse(Float32, readline())
    x = [x1 x2]

    z1 = x * W1 .+ b1
    a1 = sigmoid.(z1)
    z2 = a1 * W2 .+ b2
    a2 = sigmoid.(z2)

    println("Saída da porta lógica XOR para a entrada ($(x1), $(x2)): $(round(a2[1]; digits=3))\n")
end

 
 
# Exemplo de uso:
# julia xor.jl
# Digite os valores de entrada (0 ou 1) para a porta lógica XOR:
# 0
# 1
# Saída da porta lógica XOR para a entrada (0.0, 1.0): 1.0
# Digite os valores de entrada (0 ou 1) para a porta lógica XOR:
 