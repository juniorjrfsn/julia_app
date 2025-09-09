using Random

# Função de ativação ReLU
relu(x) = max.(0, x)

# Função de ativação Sigmoid
sigmoid(x) = 1.0 ./ (1.0 .+ exp.(-x))

# Função para normalizar os dados
function normalizar_dados(dados)
    min_vals = [minimum(dados[:, i]) for i in 1:size(dados, 2)]
    max_vals = [maximum(dados[:, i]) for i in 1:size(dados, 2)]
    dados_normalizados = similar(dados, Float32)
    for i in 1:size(dados, 2)
        dados_normalizados[:, i] = (dados[:, i] .- min_vals[i]) ./ (max_vals[i] - min_vals[i])
    end
    return dados_normalizados
end

# Função para inicializar os pesos
function inicializar_pesos(tamanho_entrada, tamanho_saida)
    return randn(Float32, tamanho_saida, tamanho_entrada) * sqrt(2.0 / tamanho_entrada)
end

# Função para inicializar os biases
function inicializar_biases(tamanho_saida)
    return zeros(Float32, tamanho_saida)
end

# Função para calcular a saída da rede neural
function calcular_saida(entrada, pesos, biases)
    camada_oculta = relu.(pesos[:camada_oculta] * entrada .+ biases[:camada_oculta])
    saida = sigmoid.(pesos[:camada_saida] * camada_oculta .+ biases[:camada_saida])
    return saida
end

# Função para treinar a rede neural
function treinar_rede(dados_treinamento, taxa_aprendizagem, epocas)
    pesos = Dict(
        :camada_oculta => inicializar_pesos(4, 5),
        :camada_saida => inicializar_pesos(5, 3)
    )
    biases = Dict(
        :camada_oculta => inicializar_biases(5),
        :camada_saida => inicializar_biases(3)
    )

    for epoca in 1:epocas
        for (entrada, saida_desejada) in dados_treinamento
            entrada = entrada'
            saida_rede = calcular_saida(entrada, pesos, biases)
            erro = saida_rede .- saida_desejada
            gradiente_saida = erro .* saida_rede .* (1 .- saida_rede)
            camada_oculta = relu.(pesos[:camada_oculta] * entrada .+ biases[:camada_oculta])
            gradiente_camada_oculta = (pesos[:camada_saida]' * gradiente_saida) .* camada_oculta .* (1 .- camada_oculta)
            pesos[:camada_saida] .-= taxa_aprendizagem * gradiente_saida * camada_oculta'
            biases[:camada_saida] .-= taxa_aprendizagem * gradiente_saida
            pesos[:camada_oculta] .-= taxa_aprendizagem * gradiente_camada_oculta * entrada'
            biases[:camada_oculta] .-= taxa_aprendizagem * gradiente_camada_oculta
        end
    end
    return pesos, biases
end

# Dados de entrada e saída de exemplo (substitua pelos seus dados)
dados = [
    ([70.0, 1.75, 30.0, 3.0], [0.6, 0.6, 0.6]), # IMC normal, atividade moderada
    ([90.0, 1.65, 40.0, 2.0], [0.8, 0.4, 0.2]), # Sobrepeso, atividade leve
    ([50.0, 1.80, 25.0, 4.0], [0.4, 0.8, 0.8])  # Abaixo do peso, atividade intensa
]

# Normaliza os dados de entrada
dados_normalizados = [(normalizar_dados(reshape(d[1], 1, :)), d[2]) for d in dados]

# Treina a rede neural
taxa_aprendizagem = 0.1
epocas = 1000
pesos, biases = treinar_rede(dados_normalizados, taxa_aprendizagem, epocas)

# Função para solicitar entradas do usuário
function obter_entradas()
    println("Insira as características de saúde:")
    println("1. Peso (kg)")
    peso = parse(Float32, readline())
    println("2. Altura (m)")
    altura = parse(Float32, readline())
    println("3. Idade (anos)")
    idade = parse(Float32, readline())
    println("4. Nível de atividade física (1-5)")
    atividade_fisica = parse(Float32, readline())

    # Normaliza as entradas do usuário
    entradas_normalizadas = normalizar_dados(reshape([peso, altura, idade, atividade_fisica], 1, :))
    return entradas_normalizadas'
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

# Loop principal
while true
    entradas_usuario = obter_entradas()
    saida_rede = calcular_saida(entradas_usuario, pesos, biases)
    (imc_msg, recomendacao_msg1, recomendacao_msg2) = interpretar_saida(saida_rede)

    println("---------------------")
    println("IMC estimado: ", saida_rede[1], " - ", imc_msg)
    println("---------------------")
    println("Recomendações de atividade física:")
    println("1. ", recomendacao_msg1)
    println("2. ", recomendacao_msg2)
end

## Execute ##
# $ julia imc4.jl