# Função sigmoide
function sigmoid(x)
    return 1.0 ./ (1.0 .+ exp.(-x))
end

# Função para solicitar entradas do usuário
function obter_entradas()
    println("Insira as características de saúde:")
    println("1. Peso (kg)")
    peso = parse(Float64, readline())
    println("2. Altura (m)")
    altura = parse(Float64, readline())
    println("3. Idade (anos)")
    idade = parse(Float64, readline())
    println("4. Nível de atividade física (1-5)")
    atividade_fisica = parse(Float64, readline())
    return [peso, altura, idade, atividade_fisica]
end

# Função para interpretar as saídas
function interpretar_saida(saidas)
    imc = saidas[1]
    recomendacao1 = saidas[2]
    recomendacao2 = saidas[3]

    if imc < 0.5
        imc_msg = "IMC baixo. Pode ser necessário ganhar peso para uma saúde ideal."
    elseif imc < 0.75
        imc_msg = "IMC normal. Mantenha um estilo de vida saudável."
    else
        imc_msg = "IMC alto. Pode ser necessário perder peso para uma saúde ideal."
    end

    if recomendacao1 < 0.5
        recomendacao_msg1 = "Nível baixo de atividade física recomendado."
    elseif recomendacao1 < 0.75
        recomendacao_msg1 = "Nível moderado de atividade física recomendado."
    else
        recomendacao_msg1 = "Nível alto de atividade física recomendado."
    end

    if recomendacao2 < 0.5
        recomendacao_msg2 = "Considere atividades físicas leves como caminhada."
    elseif recomendacao2 < 0.75
        recomendacao_msg2 = "Considere atividades físicas moderadas como ciclismo."
    else
        recomendacao_msg2 = "Considere atividades físicas intensas como corrida."
    end
    return (imc_msg, recomendacao_msg1, recomendacao_msg2)
end

# Dados de entrada
x = obter_entradas()

# Pesos e Biases
W_input_hidden = [0.1 0.2 0.3 0.4;
                  0.5 0.6 0.7 0.8;
                  0.9 1.0 1.1 1.2;
                  1.3 1.4 1.5 1.6;
                  1.7 1.8 1.9 2.0]

b_hidden = [0.1, 0.2, 0.3, 0.4, 0.5]

W_hidden_output = [0.1 0.2 0.3 0.4 0.5;
                   0.6 0.7 0.8 0.9 1.0;
                   1.1 1.2 1.3 1.4 1.5]

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

# Interpretação das saídas
(imc_msg, recomendacao_msg1, recomendacao_msg2) = interpretar_saida(output_output)
println("---------------------")
println("IMC estimado: ", output_output[1], " - ", imc_msg)
println("---------------------")
println("Recomendações de atividade física:")
println("1. ", recomendacao_msg1)
println("2. ", recomendacao_msg2)

## Execute ##
# $ julia imc.jl