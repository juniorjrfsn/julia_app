# Script principal para rodar o servidor persistentemente
using Pkg

# Ativa o ambiente local da pasta 'servidor'
Pkg.activate(dirname(@__FILE__))

# Inclui o arquivo do servidor diretamente (evita lentidão de pré-compilação do pacote)
include(joinpath(dirname(@__FILE__), "src", "servidor.jl"))

# Inicia o servidor HTTP na porta 8080
servidor.start_server(8080)


# julia servidor/main.jl