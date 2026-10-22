#!/usr/bin/env julia
# projeto: cnncheckin
# file: src/capture_iriun.jl
# Captura via webcam (Iriun, USB ou integrada). Usa src/camera.jl.
#
# Uso:
#   julia --project=.. capture_iriun.jl --list
#   julia --project=.. capture_iriun.jl --test
#   julia --project=.. capture_iriun.jl --single foto.jpg [--camera N]
#   julia --project=.. capture_iriun.jl --multiple "João" ./fotos 15 [--camera N]

include(joinpath(@__DIR__, "camera.jl"))
using .Camera

const HELP = """
USO:
  capture_iriun.jl --list
  capture_iriun.jl --test
  capture_iriun.jl --single <arquivo.jpg> [--camera N|nome]
  capture_iriun.jl --multiple <nome> <pasta> <num> [--camera N|nome]
  capture_iriun.jl --help
"""

# Remove "--camera X" de args e devolve X (padrão 0).
function take_camera!(args::Vector{String})
    i = findfirst(==("--camera"), args)
    i === nothing && return 0
    i == length(args) && error("--camera exige um valor")
    value = args[i + 1]
    deleteat!(args, i:i+1)
    return all(isdigit, value) ? parse(Int, value) : value
end

function test_cameras()
    println("Testando câmeras...")
    for idx in list_cameras()
        cam = open_camera(idx; retries=1)
        cam === nothing && continue
        ok = 0
        for _ in 1:5
            Camera.grab_frame(cam) !== nothing && (ok += 1)
            sleep(0.15)
        end
        try close(cam) catch end
        println("  [$idx] ", ok >= 4 ? "ESTÁVEL" : "INSTÁVEL", " ($ok/5 frames)")
    end
end

function main(argv)
    args = String.(argv)
    if isempty(args) || "--help" in args
        print(HELP)
        return
    end
    camera = take_camera!(args)
    cmd = args[1]

    if cmd == "--list"
        list_cameras()
    elseif cmd == "--test"
        test_cameras()
    elseif cmd == "--single"
        length(args) < 2 && (println("Erro: informe o arquivo de saída."); return)
        capture_single(args[2]; camera=camera)
    elseif cmd == "--multiple"
        length(args) < 4 && (println("Erro: --multiple <nome> <pasta> <num>"); return)
        capture_session(args[2], args[3], parse(Int, args[4]); camera=camera)
    else
        println("Comando inválido: $cmd")
        print(HELP)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__()
    main(ARGS)
end
