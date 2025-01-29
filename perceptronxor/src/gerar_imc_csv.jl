# Dados de exemplo
dados_exemplo = [
    [70.0, 1.75, 30.0, 4.0, 22.86],
    [65.0, 1.65, 25.0, 3.0, 23.94],
    [80.0, 1.80, 35.0, 5.0, 24.69],
    [55.0, 1.55, 28.0, 2.0, 22.95],
    [90.0, 1.90, 40.0, 4.0, 24.76]
]

# Converter os dados para um DataFrame
using DataFrames
df = DataFrame(peso=dados_exemplo[:, 1],
               altura=dados_exemplo[:, 2],
               idade=dados_exemplo[:, 3],
               atividade_fisica=dados_exemplo[:, 4],
               imc=dados_exemplo[:, 5])

# Escrever o DataFrame em um arquivo CSV
using CSV
CSV.write("dados_imc.csv", df)

println("O arquivo 'dados_imc.csv' foi criado com sucesso.")


# julia  gerar_imc_csv.jl