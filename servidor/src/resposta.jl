# Este arquivo gera uma resposta JSON para requisições POST.
# Ele analisa a entrada da requisição e retorna um objeto JSON com metadados e os dados recebidos.

using HTTP, JSON

function handle_post_json(req::HTTP.Request)
    # Definindo headers para permitir CORS e especificar retorno como JSON
    headers = [
        "Access-Control-Allow-Origin" => "*",
        "Access-Control-Allow-Methods" => "POST, GET, OPTIONS",
        "Access-Control-Allow-Headers" => "Content-Type",
        "Content-Type" => "application/json; charset=utf-8"
    ]
    
    # Responder imediatamente a requisições preflight (OPTIONS) do CORS
    if req.method == "OPTIONS"
        return HTTP.Response(200, headers)
    end
    
    # Obter e converter o corpo da requisição para String
    body_str = String(req.body)
    
    data = Dict()
    try
        if !isempty(body_str)
            data = JSON.parse(body_str)
        end
    catch e
        @warn "Erro ao processar JSON: $e"
    end
    
    # Gerando a resposta JSON dinamicamente
    response_data = Dict(
        "status" => "success",
        "mensagem" => "Resposta JSON gerada com sucesso pelo servidor Julia!",
        "dados_recebidos" => data,
        "timestamp" => round(Int, time()),
        "servidor_info" => Dict(
            "linguagem" => "Julia",
            "porta" => 8080,
            "metodo_requisicao" => req.method
        )
    )
    
    return HTTP.Response(200, headers, JSON.json(response_data))
end
