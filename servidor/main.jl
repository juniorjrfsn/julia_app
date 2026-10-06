# Script principal para rodar e gerenciar o servidor HTTP
using Pkg

const PID_FILE = joinpath(dirname(abspath(@__FILE__)), "server.pid")
const LOG_FILE = joinpath(dirname(abspath(@__FILE__)), "server.log")

function get_running_pid()
    if isfile(PID_FILE)
        try
            pid_str = read(PID_FILE, String)
            pid = parse(Int, strip(pid_str))
            return pid
        catch
            return nothing
        end
    end
    return nothing
end

function is_process_running(pid::Int)
    res = ccall(:kill, Cint, (Cint, Cint), pid, 0)
    if res == 0
        return true
    elseif res == -1
        err = Libc.errno()
        if err == 1 # EPERM
            return true
        end
    end
    return false
end

function start_server_background()
    pid = get_running_pid()
    if pid !== nothing && is_process_running(pid)
        println("O servidor já está rodando (PID: $pid).")
        return
    end

    if isfile(PID_FILE)
        rm(PID_FILE, force=true)
    end

    cmd = `julia $(abspath(@__FILE__)) --run-server`
    
    try
        p = run(pipeline(cmd, stdout=LOG_FILE, stderr=LOG_FILE), wait=false)
        child_pid = getpid(p)
        
        write(PID_FILE, string(child_pid))
        
        println("Servidor iniciado com sucesso em segundo plano!")
        println("PID: $child_pid")
        println("Logs salvos em: $LOG_FILE")
    catch e
        println("Erro ao iniciar o servidor: $e")
    end
end

function stop_server()
    pid = get_running_pid()
    if pid === nothing
        println("O servidor não está rodando (nenhum arquivo PID encontrado).")
        return
    end

    if !is_process_running(pid)
        println("O arquivo PID existe, mas o processo $pid não está rodando. Limpando arquivo PID...")
        rm(PID_FILE, force=true)
        return
    end

    println("Parando o servidor (PID: $pid)...")
    
    # Envia SIGTERM (15)
    ccall(:kill, Cint, (Cint, Cint), pid, 15)
    
    # Aguarda o processo terminar (até 5 segundos)
    stopped = false
    for _ in 1:50
        if !is_process_running(pid)
            stopped = true
            break
        end
        sleep(0.1)
    end

    if !stopped
        println("Processo não respondeu ao SIGTERM. Enviando SIGKILL...")
        ccall(:kill, Cint, (Cint, Cint), pid, 9)
        sleep(0.5)
    end

    rm(PID_FILE, force=true)
    println("Servidor parado com sucesso.")
end

function check_status()
    pid = get_running_pid()
    if pid !== nothing && is_process_running(pid)
        println("Servidor está rodando (PID: $pid).")
    else
        if pid !== nothing
            println("Servidor não está rodando (PID antigo: $pid no arquivo, mas processo inativo).")
        else
            println("Servidor não está rodando.")
        end
    end
end

function restart_server()
    stop_server()
    sleep(1.0)
    start_server_background()
end

function show_usage()
    println("Uso: julia main.jl [start|stop|restart|status]")
end

# Tratamento dos argumentos da linha de comando
if length(ARGS) == 0
    show_usage()
elseif ARGS[1] == "start"
    start_server_background()
elseif ARGS[1] == "stop"
    stop_server()
elseif ARGS[1] == "restart"
    restart_server()
elseif ARGS[1] == "status"
    check_status()
elseif ARGS[1] == "--run-server"
    # Comando interno para rodar em primeiro plano (usado pelo processo filho)
    Pkg.activate(dirname(abspath(@__FILE__)))
    include(joinpath(dirname(abspath(@__FILE__)), "src", "servidor.jl"))
    servidor.start_server(8080)
else
    println("Comando desconhecido: $(ARGS[1])")
    show_usage()
end