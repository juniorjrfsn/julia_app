#!/usr/bin/env julia
# projeto: cnncheckin
# file: src/cnncheckin.jl
# Reconhecimento facial com CNN (Flux), treino incremental e check-in por webcam.
#
#   julia --project=. src/cnncheckin.jl --train dados/fotos_train [--epochs N]
#   julia --project=. src/cnncheckin.jl --add [dados/fotos_new]
#   julia --project=. src/cnncheckin.jl --identify foto.jpg
#   julia --project=. src/cnncheckin.jl --checkin presenca.csv [--camera N]
#   julia --project=. src/cnncheckin.jl --capture "João" dados/fotos_train 15 [--camera N]
#   julia --project=. src/cnncheckin.jl --list-cameras

using Flux
using Images
using FileIO
using JLD2
using Statistics
using Random
using Dates
using TOML

include(joinpath(@__DIR__, "camera.jl"))
using .Camera

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

const IMG_SIZE         = (128, 128)
const BATCH_SIZE       = 8
const MAX_EPOCHS       = 30
const INCREMENTAL_EPOCHS = 15
const LEARNING_RATE    = 1.0f-4
const INCREMENTAL_LR   = 5.0f-5
const PATIENCE         = 10
const MIN_IMPROVEMENT  = 0.001
const MIN_CONFIDENCE   = 0.6          # abaixo disso → "desconhecido"
const VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

const ROOT             = normpath(joinpath(@__DIR__, ".."))
const MODEL_PATH       = joinpath(@__DIR__, "face_recognition_model.jld2")
const CONFIG_PATH      = joinpath(@__DIR__, "face_recognition_config.toml")
const MODEL_DATA_PATH  = joinpath(@__DIR__, "face_recognition_model_data.toml")
const TRAIN_PATH       = joinpath(ROOT, "dados", "fotos_train")
const INCREMENTAL_PATH = joinpath(ROOT, "dados", "fotos_new")
const AUTH_PATH        = joinpath(ROOT, "dados", "fotos_auth")

# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------

function valid_image(path::AbstractString)
    lowercase(splitext(path)[2]) in VALID_EXTENSIONS || return false
    isfile(path) && filesize(path) >= 500
end

# "joao_03.jpg" → "joao"; "João_01_20251211_120000.jpg" → "João"
person_name(filename::AbstractString) =
    String(strip(replace(splitext(basename(filename))[1], r"([-_]\d+)+$" => "")))

"Agrupa imagens de uma pasta por pessoa: nome => [caminhos]."
function scan_dataset(dir::AbstractString)
    people = Dict{String,Vector{String}}()
    isdir(dir) || return people
    for f in sort(readdir(dir))
        p = joinpath(dir, f)
        valid_image(p) || continue
        push!(get!(people, person_name(f), String[]), p)
    end
    return people
end

function normalize_image(a::Array{Float32,3})
    μ, σ = mean(a), std(a)
    return σ > 1f-6 ? (a .- μ) ./ σ : a .- μ
end

"Imagem (qualquer tipo de cor) → Array{Float32,3} (H, W, 3) normalizado."
function image_to_array(img)
    rgb = imresize(RGB.(img), IMG_SIZE)
    arr = permutedims(Float32.(channelview(rgb)), (2, 3, 1))
    return normalize_image(arr)
end

load_array(path::AbstractString) = image_to_array(load(path))

# Variações simples para aumentar o conjunto de treino.
augmented(a) = [a,
                reverse(a, dims=2),
                clamp.(a .* 1.1f0, -3f0, 3f0),
                clamp.(a .* 0.9f0, -3f0, 3f0)]

const Items = Vector{Tuple{String,Int}}   # (caminho, rótulo)

function build_tensor(items::Items; augment::Bool=false)
    xs = Array{Float32,3}[]
    ys = Int[]
    for (path, label) in items
        arr = nothing
        try
            arr = load_array(path)
        catch e
            @warn "Ignorando $path" exception=e
        end
        arr === nothing && continue
        for v in (augment ? augmented(arr) : [arr])
            push!(xs, v)
            push!(ys, label)
        end
    end
    isempty(xs) && error("Nenhuma imagem válida pôde ser carregada.")
    X = Array{Float32,4}(undef, IMG_SIZE..., 3, length(xs))
    for (i, x) in enumerate(xs)
        X[:, :, :, i] = x
    end
    return X, ys
end

"Divide por pessoa ANTES do aumento, evitando vazamento entre treino e validação."
function split_items(people::Dict{String,Vector{String}}, names::Vector{String};
                     val_fraction::Float64=0.2)
    train, val = Items(), Items()
    for (label, name) in enumerate(names)
        paths = shuffle(people[name])
        n_val = length(paths) >= 3 ? max(1, floor(Int, length(paths) * val_fraction)) : 0
        for (i, p) in enumerate(paths)
            push!(i <= n_val ? val : train, (p, label))
        end
    end
    return train, val
end

# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

conv_block(in, out, p) = (Conv((3, 3), in => out, pad=1),
                          BatchNorm(out, relu),
                          MaxPool((2, 2)),
                          Dropout(p))

function build_model(num_classes::Int)
    Chain(
        conv_block(3, 64, 0.1)...,
        conv_block(64, 128, 0.1)...,
        conv_block(128, 256, 0.15)...,
        conv_block(256, 256, 0.15)...,
        Flux.flatten,                       # 8×8×256 = 16384
        Dense(16384 => 512, relu),
        Dropout(0.4),
        Dense(512 => 256, relu),
        Dropout(0.3),
        Dense(256 => num_classes),
    )
end

function accuracy(model, X, y)
    correct = 0
    for idxs in Iterators.partition(1:length(y), 32)
        preds = Flux.onecold(model(X[:, :, :, idxs]))
        correct += sum(preds .== y[idxs])
    end
    return correct / length(y)
end

function fit!(model, train::Items, val::Items, n_classes::Int; epochs::Int, lr::Float32)
    Xtr, ytr = build_tensor(train; augment=true)
    Xval, yval = isempty(val) ? (nothing, nothing) : build_tensor(val)
    println("Treino: $(length(ytr)) amostras (com aumento) | Validação: ",
            isnothing(yval) ? 0 : length(yval))

    opt = Flux.setup(Adam(lr), model)
    best_acc, best_epoch, since, last_epoch = -1.0, 0, 0, 0
    best_state = nothing

    for epoch in 1:epochs
        last_epoch = epoch
        Flux.trainmode!(model)
        total = 0.0
        for idxs in Iterators.partition(randperm(length(ytr)), BATCH_SIZE)
            length(idxs) < 2 && continue          # BatchNorm precisa de ≥ 2
            x = Xtr[:, :, :, idxs]
            y = Flux.onehotbatch(ytr[idxs], 1:n_classes)
            loss, grads = Flux.withgradient(m -> Flux.logitcrossentropy(m(x), y), model)
            Flux.update!(opt, model, grads[1])
            total += loss * length(idxs)
        end
        Flux.testmode!(model)
        acc = isnothing(Xval) ? accuracy(model, Xtr, ytr) : accuracy(model, Xval, yval)
        println("Época $(lpad(epoch, 2)): loss=$(round(total / length(ytr), digits=4))  ",
                "acc=$(round(acc, digits=4))")

        if acc > best_acc + MIN_IMPROVEMENT
            best_acc, best_epoch, since = acc, epoch, 0
            best_state = deepcopy(Flux.state(model))
        elseif (since += 1) >= PATIENCE
            println("Parada antecipada (sem melhora há $PATIENCE épocas).")
            break
        end
    end

    best_state === nothing || Flux.loadmodel!(model, best_state)
    Flux.testmode!(model)
    return (accuracy=best_acc, best_epoch=best_epoch, epochs_trained=last_epoch)
end

# ---------------------------------------------------------------------------
# Persistência
# ---------------------------------------------------------------------------

function save_artifacts(model, names::Vector{String}, stats; extra::Dict=Dict())
    jldsave(MODEL_PATH; model_state=Flux.state(model), person_names=names)

    cfg = Dict(
        "model" => Dict("img_width" => IMG_SIZE[1], "img_height" => IMG_SIZE[2],
                        "num_classes" => length(names),
                        "model_architecture" => "CNN_FaceRecognition_v2"),
        "training" => Dict("batch_size" => BATCH_SIZE, "final_accuracy" => stats.accuracy,
                           "best_epoch" => stats.best_epoch,
                           "epochs_trained" => stats.epochs_trained),
        "data" => Dict("person_names" => names, "data_path" => TRAIN_PATH,
                       "timestamp" => string(now())),
        "metadata" => Dict("created_by" => "CNNCheckin v2.1",
                           "last_saved" => string(now())),
    )
    merge!(cfg, extra)
    open(io -> TOML.print(io, cfg), CONFIG_PATH, "w")

    meta = Dict(
        "person_mappings" => Dict(n => i for (i, n) in enumerate(names)),
        "model_summary" => Dict("input_shape" => collect(IMG_SIZE),
                                "output_classes" => length(names),
                                "total_parameters" => sum(length, Flux.trainables(model)),
                                "created_at" => string(now())),
    )
    open(io -> TOML.print(io, meta), MODEL_DATA_PATH, "w")
end

function load_artifacts()
    isfile(MODEL_PATH) || error("Modelo não encontrado. Treine antes: --train dados/fotos_train")
    data = JLD2.load(MODEL_PATH)
    names = Vector{String}(data["person_names"])
    model = build_model(length(names))
    Flux.loadmodel!(model, data["model_state"])
    Flux.testmode!(model)
    return model, names
end

# ---------------------------------------------------------------------------
# Comandos
# ---------------------------------------------------------------------------

function cmd_train(dir; epochs::Int=MAX_EPOCHS)
    people = scan_dataset(dir)
    isempty(people) && error("Nenhuma imagem válida em: $dir")
    names = sort(collect(keys(people)))
    length(names) >= 2 || @warn "Só há 1 pessoa; o modelo não conseguirá discriminar."
    println("Pessoas: ", join(names, ", "))

    train, val = split_items(people, names)
    model = build_model(length(names))
    stats = fit!(model, train, val, length(names); epochs=epochs, lr=LEARNING_RATE)
    save_artifacts(model, names, stats)
    println("Modelo salvo em $MODEL_PATH (melhor acc=$(round(stats.accuracy, digits=4)) na época $(stats.best_epoch))")
end

"Aprendizado incremental: adiciona pessoas de `new_dir`, reaproveitando o modelo atual."
function cmd_add(new_dir)
    old_model, old_names = load_artifacts()
    new_people = scan_dataset(new_dir)
    isempty(new_people) && error("Nenhuma imagem válida em: $new_dir")

    # Mantém as imagens antigas (replay) para não esquecer quem já conhece.
    all_people = mergewith(vcat, scan_dataset(TRAIN_PATH), new_people)
    added = sort([n for n in keys(new_people) if !(n in old_names)])
    names = vcat(old_names, added)
    for n in keys(all_people)
        n in names || push!(names, n)
    end
    println("Pessoas novas: ", isempty(added) ? "(nenhuma — apenas reforço)" : join(added, ", "))

    model = build_model(length(names))
    Flux.loadmodel!(model[1:end-1], old_model[1:end-1])     # corpo da rede
    old_head, head = old_model[end], model[end]             # camada de saída ampliada
    head.weight[1:length(old_names), :] .= old_head.weight
    head.bias[1:length(old_names)] .= old_head.bias

    train, val = split_items(all_people, names)
    stats = fit!(model, train, val, length(names); epochs=INCREMENTAL_EPOCHS, lr=INCREMENTAL_LR)
    save_artifacts(model, names, stats;
                   extra=Dict("incremental_stats" => Dict("new_people" => added,
                                                          "original_people" => old_names,
                                                          "last_incremental_training" => string(now()))))

    # Move as fotos novas para a base de treino.
    if abspath(new_dir) != abspath(TRAIN_PATH)
        mkpath(TRAIN_PATH)
        for ps in values(new_people), p in ps
            mv(p, joinpath(TRAIN_PATH, basename(p)); force=true)
        end
        println("Fotos novas movidas para $TRAIN_PATH")
    end
end

function predict(model, names, arr)
    probs = softmax(vec(model(reshape(arr, size(arr)..., 1))))
    i = argmax(probs)
    return names[i], Float64(probs[i])
end

function cmd_identify(path)
    isfile(path) || error("Arquivo não encontrado: $path")
    model, names = load_artifacts()
    person, conf = predict(model, names, load_array(path))
    pct = round(conf * 100, digits=2)
    if conf < MIN_CONFIDENCE
        println("Desconhecido (melhor palpite: $person, confiança $pct%)")
    else
        println("Pessoa: $person | confiança: $pct%")
    end
end

function cmd_checkin(csv; camera=0)
    model, names = load_artifacts()
    present = Set{String}()
    today = Dates.format(today_date(), "yyyy-mm-dd")

    if isfile(csv)
        for line in Iterators.drop(eachline(csv), 1)
            f = split(line, ",")
            length(f) >= 3 && startswith(f[1], today) || continue
            f[3] == "CHECK-IN" ? push!(present, f[2]) : delete!(present, f[2])
        end
    else
        mkpath(dirname(abspath(csv)))
        write(csv, "timestamp,pessoa,acao,confianca\n")
    end

    cam = open_camera(camera)
    cam === nothing && return
    println("ENTER = registrar | 'sair' = encerrar")
    try
        while true
            print("> ")
            line = readline()
            lowercase(strip(line)) in ("sair", "q", "exit") && break
            frame = nothing
            for _ in 1:4
                frame = grab_frame(cam)      # descarta frames antigos do buffer
            end
            frame === nothing && (println("Falha ao capturar."); continue)

            person, conf = predict(model, names, image_to_array(frame))
            if conf < MIN_CONFIDENCE
                println("Não reconhecido (melhor palpite: $person, $(round(conf*100, digits=1))%)")
                continue
            end
            action = person in present ? "CHECK-OUT" : "CHECK-IN"
            action == "CHECK-IN" ? push!(present, person) : delete!(present, person)
            stamp = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
            open(io -> println(io, "$stamp,$person,$action,$(round(conf*100, digits=1))"), csv, "a")
            println("$action: $person ($(round(conf*100, digits=1))%)")
        end
    finally
        try close(cam) catch end
    end
end
today_date() = Dates.today()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

const HELP = """
Uso:
  cnncheckin.jl --train <pasta> [--epochs N]
  cnncheckin.jl --add [pasta_novas]            (padrão: dados/fotos_new)
  cnncheckin.jl --identify <imagem>
  cnncheckin.jl --checkin <arquivo.csv> [--camera N]
  cnncheckin.jl --capture <pessoa> <pasta> [num] [--camera N]
  cnncheckin.jl --list-cameras
  cnncheckin.jl --help
"""

function take_option!(args::Vector{String}, flag::String, default)
    i = findfirst(==(flag), args)
    i === nothing && return default
    i == length(args) && error("$flag exige um valor")
    value = args[i + 1]
    deleteat!(args, i:i+1)
    return value
end

function main(argv)
    args = String.(argv)
    if isempty(args) || "--help" in args
        print(HELP)
        return
    end

    cam_raw = take_option!(args, "--camera", "0")
    camera = all(isdigit, cam_raw) ? parse(Int, cam_raw) : cam_raw
    epochs = parse(Int, take_option!(args, "--epochs", string(MAX_EPOCHS)))
    cmd = args[1]

    try
        if cmd == "--list-cameras"
            list_cameras()
        elseif cmd == "--train"
            length(args) >= 2 || error("informe a pasta de treinamento")
            cmd_train(args[2]; epochs=epochs)
        elseif cmd == "--add"
            cmd_add(length(args) >= 2 ? args[2] : INCREMENTAL_PATH)
        elseif cmd == "--identify"
            length(args) >= 2 || error("informe a imagem")
            cmd_identify(args[2])
        elseif cmd == "--checkin"
            length(args) >= 2 || error("informe o arquivo CSV")
            cmd_checkin(args[2]; camera=camera)
        elseif cmd == "--capture"
            length(args) >= 3 || error("use --capture <pessoa> <pasta> [num]")
            n = length(args) >= 4 ? parse(Int, args[4]) : 10
            capture_session(args[2], args[3], n; camera=camera)
        else
            println("Opção inválida: $cmd")
            print(HELP)
        end
    catch e
        e isa InterruptException && rethrow()
        println("Erro: ", sprint(showerror, e))
        exit(1)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__()
    main(ARGS)
end
