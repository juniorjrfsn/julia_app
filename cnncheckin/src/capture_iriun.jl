#!/usr/bin/env julia
# Captura específica para Iriun Webcam

# projeto: cnncheckin
# file: cnncheckin/src/capture_iriun.jl

# Captura específica para Iriun Webcam
# Compatível com celulares Android/iOS via Iriun

 
# capture.jl
# Captura de fotos via webcam (qualquer câmera USB/integrada)
 
#
# Uso:
#   julia capture.jl --single foto.jpg
#   julia capture.jl --multiple "João" ./fotos 15
#   julia capture.jl --test
#   julia capture.jl --list

using VideoIO
using Images
using FileIO
using Dates

# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

"""
    list_cameras() -> Vector{Int}

Lista todas as câmeras disponíveis no sistema.
"""
function list_cameras()
    println("\nListando câmeras disponíveis...")
    available = Int[]

    for i in 0:10
        try
            cam = VideoIO.opencamera(i)
            if cam !== nothing
                frame = read(cam)
                if frame !== nothing
                    h, w = size(frame)[1:2]
                    push!(available, i)
                    println("  [$i] → $(w)×$(h)")
                end
                close(cam)
            end
        catch
            continue
        end
    end

    if isempty(available)
        println("  Nenhuma câmera detectada.")
    else
        println("  Total: $(length(available)) câmera(s).")
    end
    return available
end

"""
    open_camera(idx::Int; retries::Int=3) -> Union{VideoIO.Camera, Nothing}

Tenta abrir a câmera com retries.
"""
function open_camera(idx::Int; retries::Int=3)
    for attempt in 1:retries
        try
            println("   Tentativa $attempt/$retries: abrindo câmera $idx...")
            cam = VideoIO.opencamera(idx)
            if cam !== nothing && read(cam) !== nothing
                println("   Câmera $idx aberta com sucesso!")
                return cam
            end
            close(cam)
        catch e
            println("   Falha: $e")
            attempt < retries && sleep(1)
        end
    end
    println("   Não foi possível abrir a câmera $idx.")
    return nothing
end

# ---------------------------------------------------------------------------
# Captura única
# ---------------------------------------------------------------------------

"""
    capture_single(output_path::String; camera_idx::Union{Int,Nothing}=nothing, countdown::Int=3) -> Bool
"""
function capture_single(output_path::String; camera_idx::Union{Int,Nothing}=nothing, countdown::Int=3)
    println("\nCAPTURA ÚNICA")
    println("="^50)

    mkpath(dirname(output_path))

    # Selecionar câmera
    if camera_idx === nothing
        cams = list_cameras()
        if isempty(cams)
            return false
        end
        camera_idx = cams[1]
        println("Usando câmera padrão: [$camera_idx]")
    end

    cam = open_camera(camera_idx)
    cam === nothing && return false

    # Contagem regressiva
    if countdown > 0
        println("\nPosicione-se... Captura em $countdown segundo(s):")
        for i in countdown:-1:1
            print("   $i ")
            flush(stdout)
            try read(cam) catch end
            sleep(1)
        end
        println()
    end

    # Capturar
    println("Capturando...")
    frame = read(cam)
    close(cam)

    if frame === nothing
        println("Falha ao capturar frame.")
        return false
    end

    img = RGB.(frame)
    save(output_path, img)
    w, h = size(img)
    println("Salvo: $output_path ($w×$h)")
    return true
end

# ---------------------------------------------------------------------------
# Captura múltipla
# ---------------------------------------------------------------------------

"""
    capture_multiple(person_name::String, output_dir::String, n::Int=15;
                     camera_idx::Union{Int,Nothing}=nothing, delay::Int=2) -> Int
"""
function capture_multiple(person_name::String, output_dir::String, n::Int=15;
                          camera_idx::Union{Int,Nothing}=nothing, delay::Int=2)
    println("\nCAPTURA MÚLTIPLA")
    println("="^50)
    println("   Pessoa: $person_name")
    println("   Imagens: $n")
    println("   Intervalo: $delay s")
    println("   Pasta: $output_dir")
    println("="^50)

    mkpath(output_dir)

    # Selecionar câmera
    if camera_idx === nothing
        cams = list_cameras()
        if isempty(cams)
            return 0
        end
        camera_idx = cams[1]
        println("Usando câmera padrão: [$camera_idx]")
    end

    cam = open_camera(camera_idx)
    cam === nothing && return 0

    println("\nIniciando... Varie ângulo e expressão!\n")
    captured = 0

    for i in 1:n
        println("[$i/$n] Preparando...")

        for j in delay:-1:1
            print("   $j ")
            flush(stdout)
            try read(cam) catch end
            sleep(1)
        end
        println()

        frame = read(cam)
        if frame === nothing
            println("   Falha no frame $i")
            continue
        end

        ts = Dates.format(now(), "yyyymmdd_HHMMSS")
        fname = "$(person_name)_$(lpad(i, 2, '0'))_$ts.jpg"
        fpath = joinpath(output_dir, fname)

        img = RGB.(frame)
        save(fpath, img)
        captured += 1
        println("   Salvo: $fname")
    end

    try close(cam) catch end

    rate = round(captured/n * 100, digits=1)
    println("\n" * "="^50)
    println("CONCLUÍDO: $captured/$n imagens ($rate%)")
    if captured >= n * 0.75
        println("Suficiente para treinamento!")
    else
        println("Recomendado: repetir captura.")
    end
    println("="^50)

    return captured
end

# ---------------------------------------------------------------------------
# Teste de câmeras
# ---------------------------------------------------------------------------

"""
    test_cameras()
"""
function test_cameras()
    println("\nTESTE DE CÂMERAS")
    println("="^50)

    cams = list_cameras()
    isempty(cams) && return

    println("\nTestando estabilidade (5 frames cada)...\n")
    for idx in cams
        println("─"^50)
        println("Câmera $idx")
        cam = open_camera(idx; retries=1)
        cam === nothing && continue

        ok = 0
        for _ in 1:5
            if read(cam) !== nothing
                ok += 1
            end
            sleep(0.15)
        end
        close(cam)

        status = ok >= 4 ? "ESTÁVEL" : "INSTÁVEL"
        println("   → $status ($ok/5 frames)")
    end
    println("="^50)
end

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

function print_help()
    println("""
    USO:
      julia capture.jl --single <arquivo.jpg> [--camera N]
      julia capture.jl --multiple <nome> <pasta> <num> [--camera N]
      julia capture.jl --test
      julia capture.jl --list
      julia capture.jl --help

    COMANDOS:
      --single     Captura 1 foto
      --multiple   Captura várias fotos
      --test       Testa todas as câmeras
      --list       Lista câmeras
      --help       Esta ajuda

    OPÇÕES:
      --camera N   Usa câmera específica (índice)

    EXEMPLOS:
      julia capture.jl --list
      julia capture.jl --test
      julia capture.jl --single foto_teste.jpg
      julia capture.jl --single foto.jpg --camera 1
      julia capture.jl --multiple "Ana" ./fotos 20
    """)
end

function main()
    if isempty(ARGS) || "--help" in ARGS
        print_help()
        return
    end

    cmd = ARGS[1]

    if cmd == "--list"
        list_cameras()

    elseif cmd == "--test"
        test_cameras()

    elseif cmd == "--single"
        length(ARGS) < 2 && (println("Erro: informe o caminho do arquivo."); print_help(); return)
        out = ARGS[2]
        cam = nothing
        if length(ARGS) >= 4 && ARGS[3] == "--camera"
            cam = parse(Int, ARGS[4])
        end
        capture_single(out; camera_idx=cam)

    elseif cmd == "--multiple"
        length(ARGS) < 4 && (println("Erro: uso: --multiple <nome> <pasta> <num> [--camera N]"); print_help(); return)
        name = ARGS[2]
        dir = ARGS[3]
        num = parse(Int, ARGS[4])
        cam = nothing
        if length(ARGS) >= 6 && ARGS[5] == "--camera"
            cam = parse(Int, ARGS[6])
        end
        capture_multiple(name, dir, num; camera_idx=cam)

    else
        println("Comando inválido: $cmd")
        print_help()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end