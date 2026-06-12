module servidor
    using HTTP, Sockets, JSON
    greet() = print("Hello World!")
    #import Pkg; Pkg.add("HTTP")


    const ROUTER = HTTP.Router()

    function getItems(req::HTTP.Request)
        headers = [
            "Access-Control-Allow-Origin" => "*",
            "Access-Control-Allow-Methods" => "GET, OPTIONS"
        ]
        if req.method == "OPTIONS"
            return HTTP.Response(200, headers)
        end
        return HTTP.Response(200, headers, JSON.json(rand(2)))
    end

    function events(stream::HTTP.Stream)
        HTTP.setheader(stream, "Access-Control-Allow-Origin" => "*")
        HTTP.setheader(stream, "Access-Control-Allow-Methods" => "GET, OPTIONS")
        HTTP.setheader(stream, "Content-Type" => "text/event-stream")

        if stream.message.method == "OPTIONS"
            return nothing
        end

        HTTP.setheader(stream, "Content-Type" => "text/event-stream")
        HTTP.setheader(stream, "Cache-Control" => "no-cache")
        while true
            write(stream, "event: ping\ndata: $(round(Int, time()))\n\n")
            if rand(Bool)
                write(stream, "data: $(rand())\n\n")
            end
            sleep(1)
        end
        return nothing
    end

    # Inclui o arquivo de resposta JSON
    include("resposta.jl")

    # Função para servir a página HTML
    function serve_html(req::HTTP.Request)
        headers = ["Content-Type" => "text/html; charset=utf-8"]
        html_path = joinpath(dirname(@__FILE__), "..", "public", "index.html")
        if isfile(html_path)
            return HTTP.Response(200, headers, read(html_path, String))
        else
            return HTTP.Response(404, headers, "<h1>404 - Arquivo index.html não encontrado</h1><p>Caminho: $html_path</p>")
        end
    end

    # Registro das rotas
    HTTP.register!(ROUTER, "GET", "/", HTTP.streamhandler(serve_html))
    HTTP.register!(ROUTER, "GET", "/api/getItems", HTTP.streamhandler(getItems))
    HTTP.register!(ROUTER, "POST", "/api/postJSON", HTTP.streamhandler(handle_post_json))
    HTTP.register!(ROUTER, "OPTIONS", "/api/postJSON", HTTP.streamhandler(handle_post_json))
    HTTP.register!(ROUTER, "/api/events", events)

    # Função para iniciar o servidor de forma síncrona/persistente
    function start_server(port=8080)
        println("Servidor HTTP Julia ativo em http://127.0.0.1:$port")
        println("Abra http://127.0.0.1:$port em seu navegador para testar a requisição jQuery!")
        HTTP.listen(ROUTER, "127.0.0.1", port)
    end

    # Autoteste assíncrono mantido para compatibilidade
    server = HTTP.listen!(ROUTER, "127.0.0.1", 8080)

    # Julia usage
    resp = HTTP.get("http://localhost:8080/api/getItems")

    should_close = Ref(false)
    @async HTTP.open("GET", "http://127.0.0.1:8080/api/events") do io
        while !eof(io) && !should_close[]
            println(String(readavailable(io)))
        end
    end

    # run the following to stop the streaming client request
    should_close[] = true

    # close the server which will stop the HTTP server from listening
    close(server)
    @assert istaskdone(server.serve_task)
end # module servidor

