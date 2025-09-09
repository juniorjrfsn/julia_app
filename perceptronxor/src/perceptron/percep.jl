# Função sigmoide
function sigmoid(x)
    return 1.0 ./ (1.0 .+ exp.(-x))
end

# Entradas
x = [0.5, 0.3, 0.2]

# Pesos e Biases
W_input_hidden = [0.1 0.2 0.3;
                  0.4 0.5 0.6;
                  0.7 0.8 0.9]

b_hidden = [0.1, 0.2, 0.3]

W_hidden_output = [0.1 0.2 0.3;
                   0.4 0.5 0.6;
                   0.7 0.8 0.9]

b_output = [0.1, 0.2, 0.3]

# Forward pass
# Camada Oculta
hidden_input = W_input_hidden * x .+ b_hidden
hidden_output = sigmoid(hidden_input)

# Camada de Saída
output_input = W_hidden_output * hidden_output .+ b_output
output_output = sigmoid(output_input)

# Resultado
println("Ativações na Camada Oculta: ", hidden_output)
println("Saídas: ", output_output)

## Execute ##
# $ cd .\perceptronxor\src\
# $ julia percep.jl