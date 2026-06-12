function seed_demo_tasks!()
    if isempty(DB)
        create_task("Estudar Julia MVC")
        create_task("Construir uma API REST")
        create_task("Deploy no servidor")
    end
end

function start(; host::String = "127.0.0.1", port::Int = 9090)
    seed_demo_tasks!()
    @info "🚀 Servidor iniciado em http://$(host):$(port)"
    HTTP.serve(route, host, port)
end
