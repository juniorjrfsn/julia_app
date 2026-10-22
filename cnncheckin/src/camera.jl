# projeto: cnncheckin
# file: src/camera.jl
# Utilitários de câmera compartilhados por cnncheckin.jl e capture_iriun.jl

module Camera

using VideoIO
using Images
using FileIO
using Dates

export camera_device, open_camera, grab_frame, list_cameras,
       capture_single, capture_session

"""
    camera_device(spec) -> String

Converte um índice (0, 1, ...) ou nome de dispositivo no identificador que o
VideoIO espera, conforme o sistema operacional.
"""
function camera_device(spec)
    if spec isa AbstractString && !all(isdigit, spec)
        return String(spec)                      # nome/caminho explícito
    end
    idx = spec isa Integer ? Int(spec) : parse(Int, spec)
    if Sys.islinux()
        return "/dev/video$idx"
    elseif Sys.iswindows()
        isdefined(VideoIO, :init_camera_devices) &&
            try VideoIO.init_camera_devices() catch end
        devs = isdefined(VideoIO, :CAMERA_DEVICES) ?
               getfield(VideoIO, :CAMERA_DEVICES) : String[]
        return idx + 1 <= length(devs) ? devs[idx + 1] :
               VideoIO.DEFAULT_CAMERA_DEVICE[]
    else
        return string(idx)                       # macOS (avfoundation)
    end
end

"Lê um frame; devolve `nothing` em caso de erro."
function grab_frame(cam)
    try
        return read(cam)
    catch
        return nothing
    end
end

"Abre a câmera (com tentativas) e descarta alguns frames de aquecimento."
function open_camera(spec; retries::Int=3)
    dev = camera_device(spec)
    for attempt in 1:retries
        try
            cam = VideoIO.opencamera(dev)
            for _ in 1:3
                grab_frame(cam)
            end
            if grab_frame(cam) !== nothing
                return cam
            end
            close(cam)
        catch e
            println("  tentativa $attempt/$retries falhou: $(sprint(showerror, e))")
        end
        attempt < retries && sleep(1)
    end
    println("  não foi possível abrir a câmera $spec ($dev)")
    return nothing
end

"Lista câmeras que respondem; devolve os índices."
function list_cameras(; max_index::Int=6)
    found = Int[]
    for i in 0:max_index-1
        cam = try open_camera(i; retries=1) catch; nothing end
        cam === nothing && continue
        frame = grab_frame(cam)
        if frame !== nothing
            h, w = size(frame)
            push!(found, i)
            println("  [$i] → $(w)×$(h)")
        end
        try close(cam) catch end
    end
    isempty(found) && println("  nenhuma câmera detectada")
    return found
end

# Espera `seconds` segundos mantendo o buffer da câmera fluindo.
function countdown!(cam, seconds::Int)
    for i in seconds:-1:1
        print("  $i ")
        flush(stdout)
        t0 = time()
        while time() - t0 < 1.0
            grab_frame(cam)
            sleep(0.05)
        end
    end
    println()
end

"Captura uma foto e salva em `path`."
function capture_single(path::AbstractString; camera=0, countdown::Int=3)
    mkpath(dirname(abspath(path)))
    cam = open_camera(camera)
    cam === nothing && return false
    countdown > 0 && (println("Posicione-se..."); countdown!(cam, countdown))
    frame = grab_frame(cam)
    try close(cam) catch end
    if frame === nothing
        println("Falha ao capturar o frame.")
        return false
    end
    save(path, RGB.(frame))
    h, w = size(frame)
    println("Salvo: $path ($(w)×$(h))")
    return true
end

"Captura `n` fotos de `person` em `dir`. Devolve quantas foram salvas."
function capture_session(person::AbstractString, dir::AbstractString, n::Int=15;
                         camera=0, delay::Int=2)
    mkpath(dir)
    cam = open_camera(camera)
    cam === nothing && return 0
    println("Capturando $n imagens de \"$person\". Varie ângulo e expressão!")
    captured = 0
    for i in 1:n
        println("[$i/$n]")
        countdown!(cam, delay)
        frame = grab_frame(cam)
        if frame === nothing
            println("  falha no frame $i")
            continue
        end
        ts = Dates.format(now(), "yyyymmdd_HHMMSS")
        name = "$(person)_$(lpad(i, 2, '0'))_$ts.jpg"
        save(joinpath(dir, name), RGB.(frame))
        captured += 1
        println("  salvo: $name")
    end
    try close(cam) catch end
    println("Concluído: $captured/$n imagens.")
    return captured
end

end # module
