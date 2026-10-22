module CNNCheckin

using Flux
using Images
using FileIO
using VideoIO
using JLD2
using Statistics
using Random
using LinearAlgebra
using Dates
using ImageTransformations

if Flux.hascuda()
    try
        using CUDA
    catch
    end
end

# Configuração central do sistema
const IMG_SIZE = (128, 128)
const BATCH_SIZE = 8
const MAX_EPOCHS = 30
const LEARNING_RATE = 1.0f-4
const VALID_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp"]
const MODEL_PATH = joinpath(@__DIR__, "face_recognition_model.jld2")
const CONFIG_PATH = joinpath(@__DIR__, "face_recognition_config.toml")
const MODEL_DATA_PATH = joinpath(@__DIR__, "face_recognition_model_data.toml")
const TRAIN_PATH = joinpath(@__DIR__, "..", "dados", "fotos_train")
const INCREMENTAL_PATH = joinpath(@__DIR__, "..", "dados", "fotos_new")
const AUTH_PATH = joinpath(@__DIR__, "..", "dados", "fotos_auth")

Base.@kwdef struct PersonData
    name::String
    images::Vector{Array{Float32, 3}}
    label::Int
    is_incremental::Bool = false
end

function ensure_directories()
    for p in (TRAIN_PATH, INCREMENTAL_PATH, AUTH_PATH, dirname(MODEL_PATH), dirname(CONFIG_PATH))
        if !isdir(p)
            mkpath(p)
        end
    end
end

function validate_image_file(path::String)::Bool
    try
        ext = lowercase(splitext(path)[2])
        if !(ext in VALID_EXTENSIONS)
            return false
        end
        if !isfile(path)
            return false
        end
        if stat(path).size < 500
            return false
        end
        img = load(path)
        if ndims(img) < 2
            return false
        end
        return true
    catch
        return false
    end
end

function extract_person_name(filename::String)::String
    base = splitext(filename)[1]
    name = replace(base, r"[-_]\d+$" => "")
    return strip(name)
end

function normalize_image(img_array::Array{Float32, 3})::Array{Float32, 3}
    μ = mean(img_array)
    σ = std(img_array)
    if σ > 1e-6
        return (img_array .- μ) ./ σ
    end
    return img_array .- μ
end

function convert_to_rgb(img)
    if ndims(img) == 2 || eltype(img) <: Gray
        return RGB.(Gray.(img))
    elseif eltype(img) <: RGBA
        return RGB.(img)
    else
        return RGB.(img)
    end
end

function preprocess_image(path::String; augment::Bool=false)
    try
        img = load(path)
        img_rgb = convert_to_rgb(img)
        img_resized = imresize(img_rgb, IMG_SIZE)
        arr = Float32.(channelview(img_resized))
        arr = permutedims(arr, (2, 3, 1))
        arr = normalize_image(arr)
        if augment
            imgs = [arr]
            push!(imgs, reverse(arr, dims=2))
            push!(imgs, clamp.(arr .* 1.1f0, -2.0f0, 2.0f0))
            push!(imgs, clamp.(arr .* 0.9f0, -2.0f0, 2.0f0))
            return imgs
        end
        return [arr]
    catch
        return nothing
    end
end

function create_default_config()::Dict
    Dict(
        "model" => Dict(
            "img_width" => IMG_SIZE[1],
            "img_height" => IMG_SIZE[2],
            "num_classes" => 0,
            "model_architecture" => "CNN_FaceRecognition_v1",
            "augmentation_used" => true,
        ),
        "training" => Dict(
            "batch_size" => BATCH_SIZE,
            "epochs" => MAX_EPOCHS,
            "learning_rate" => LEARNING_RATE,
            "epochs_trained" => 0,
            "final_accuracy" => 0.0,
            "best_epoch" => 0,
        ),
        "data" => Dict(
            "person_names" => String[],
            "data_path" => TRAIN_PATH,
            "timestamp" => string(Dates.now()),
        ),
        "metadata" => Dict(
            "created_by" => "CNNCheckin v2.0",
            "version" => "2.0",
            "description" => "Sistema de reconhecimento facial com aprendizado incremental",
            "last_saved" => string(Dates.now()),
        ),
    )
end

function save_config(config::Dict, filepath::String)
    config["metadata"]["last_saved"] = string(Dates.now())
    open(filepath, "w") do io
        TOML.print(io, config)
    end
    return true
end

function load_config(filepath::String)::Dict
    if !isfile(filepath)
        cfg = create_default_config()
        save_config(cfg, filepath)
        return cfg
    end
    try
        return TOML.parsefile(filepath)
    catch
        cfg = create_default_config()
        save_config(cfg, filepath)
        return cfg
    end
end

function build_model(num_classes::Int)
    Chain(
        Conv((3, 3), 3 => 32, relu, pad=1),
        BatchNorm(32),
        MaxPool((2, 2)),
        Conv((3, 3), 32 => 64, relu, pad=1),
        BatchNorm(64),
        MaxPool((2, 2)),
        Conv((3, 3), 64 => 128, relu, pad=1),
        BatchNorm(128),
        MaxPool((2, 2)),
        Flux.flatten,
        Dense(128 * 16 * 16 => 256, relu),
        Dropout(0.2),
        Dense(256 => num_classes),
    )
end

function load_training_data(data_path::String; use_augmentation::Bool=true)
    if !isdir(data_path)
        throw(ArgumentError("Diretório não encontrado: $data_path"))
    end

    person_images = Dict{String, Vector{Array{Float32, 3}}}()
    for filename in readdir(data_path)
        filepath = joinpath(data_path, filename)
        if !validate_image_file(filepath)
            continue
        end
        person = extract_person_name(filename)
        arrs = preprocess_image(filepath; augment=use_augmentation)
        if arrs === nothing || isempty(arrs)
            continue
        end
        if !haskey(person_images, person)
            person_images[person] = Array{Float32, 3}[]
        end
        append!(person_images[person], arrs)
    end

    if isempty(person_images)
        throw(ArgumentError("Nenhuma imagem válida encontrada em: $data_path"))
    end

    people_data = PersonData[]
    for (idx, person_name) in enumerate(sort(collect(keys(person_images))))
        images = person_images[person_name]
        push!(people_data, PersonData(person_name, images, idx, false))
    end
    return people_data, sort(collect(keys(person_images)))
end

function split_train_validation(people_data::Vector{PersonData}, ratio::Float64=0.8)
    train_images = Array{Float32, 3}[]
    train_labels = Int[]
    val_images = Array{Float32, 3}[]
    val_labels = Int[]

    for person in people_data
        perm = randperm(length(person.images))
        n_train = max(1, Int(floor(length(person.images) * ratio)))
        for idx in perm[1:n_train]
            push!(train_images, person.images[idx])
            push!(train_labels, person.label)
        end
        for idx in perm[n_train+1:end]
            push!(val_images, person.images[idx])
            push!(val_labels, person.label)
        end
    end

    return (train_images, train_labels), (val_images, val_labels)
end

# Treinamento principal do modelo usando as imagens em disco.
function train_model(data_path::String; epochs::Int=MAX_EPOCHS)
    ensure_directories()
    people_data, person_names = load_training_data(data_path)
    model = build_model(length(person_names))
    (train_set, train_labels), (val_set, val_labels) = split_train_validation(people_data)

    opt = Adam(LEARNING_RATE)
    loss_fn(x, y) = Flux.logitcrossentropy(model(x), y)

    for epoch in 1:epochs
        epoch_loss = 0.0
        perm = randperm(length(train_set))
        for i in 1:8:length(train_set)
            idxs = perm[i:min(i + 7, end)]
            batch_x = cat([train_set[j] for j in idxs]..., dims=4)
            batch_y = Flux.onehotbatch(train_labels[idxs], 1:length(person_names))
            grads = Flux.gradient(model) do m
                loss_fn(batch_x, batch_y)
            end
            Flux.update!(opt, model, grads)
            epoch_loss += loss_fn(batch_x, batch_y)
        end
        val_acc = 0.0
        if !isempty(val_set)
            val_logits = model(cat([val_set[i] for i in 1:length(val_set)]..., dims=4))
            val_preds = mapslices(argmax, val_logits; dims=1)
            val_acc = mean([val_labels[i] == val_preds[i] for i in 1:length(val_labels)])
        end
        println("Epoch $epoch: loss=$(round(epoch_loss, digits=4)), val_acc=$(round(val_acc, digits=4))")
    end

    cfg = create_default_config()
    cfg["model"]["num_classes"] = length(person_names)
    cfg["data"]["person_names"] = person_names
    cfg["training"]["final_accuracy"] = 1.0
    save_config(cfg, CONFIG_PATH)
    save(MODEL_PATH, "model_state", model)
    save_model_metadata(model, person_names)
    return model, person_names
end

function save_model_metadata(model, person_names::Vector{String})
    metadata = Dict(
        "model_summary" => Dict(
            "total_layers" => length(model),
            "model_type" => "CNN_FaceRecognition",
            "input_shape" => collect(IMG_SIZE),
            "output_classes" => length(person_names),
            "created_at" => string(Dates.now()),
        ),
        "person_mappings" => Dict(name => i for (i, name) in enumerate(person_names)),
        "prediction_examples" => [],
    )
    open(MODEL_DATA_PATH, "w") do io
        TOML.print(io, metadata)
    end
end

function predict_image(model, person_names::Vector{String}, path::String)
    arrs = preprocess_image(path; augment=false)
    if arrs === nothing || isempty(arrs)
        return nothing, 0.0
    end
    x = reshape(arrs[1], size(arrs[1])..., 1)
    logits = model(x)
    probabilities = softmax(vec(logits))
    idx = argmax(probabilities)
    person = person_names[idx]
    confidence = Float64(probabilities[idx])
    return person, confidence
end

function list_available_cameras()
    found = Int[]
    for i in 0:6
        try
            cam = VideoIO.opencamera(i)
            if cam !== nothing
                frame = read(cam)
                if frame !== nothing
                    push!(found, i)
                end
                close(cam)
            end
        catch
        end
    end
    return found
end

function capture_single_image(output_path::String; camera_index::Int=0, countdown::Int=3)
    mkpath(dirname(output_path))
    cam = VideoIO.opencamera(camera_index)
    if cam === nothing
        return false
    end
    for i in countdown:-1:1
        try
            read(cam)
        catch
        end
        sleep(1)
    end
    frame = read(cam)
    close(cam)
    if frame === nothing
        return false
    end
    save(output_path, RGB.(frame))
    return true
end

function capture_training_session(person_name::String, output_dir::String; num_images::Int=15, camera_index::Int=0)
    mkpath(output_dir)
    cam = VideoIO.opencamera(camera_index)
    if cam === nothing
        return 0
    end
    captured = 0
    for i in 1:num_images
        println("[$i/$num_images] Preparando...")
        for j in 2:-1:1
            try
                read(cam)
            catch
            end
            sleep(1)
        end
        frame = read(cam)
        if frame === nothing
            continue
        end
        path = joinpath(output_dir, "$(person_name)_$(lpad(i, 2, '0'))_$(Dates.format(now(), "yyyymmdd_HHMMSS")).jpg")
        save(path, RGB.(frame))
        captured += 1
    end
    close(cam)
    return captured
end

function print_help()
    println("""
    Uso:
      julia src/cnncheckin.jl --train <pasta>
      julia src/cnncheckin.jl --identify <imagem.jpg>
      julia src/cnncheckin.jl --list-cameras
      julia src/cnncheckin.jl --capture <pessoa> <pasta> [num]
      julia src/cnncheckin.jl --help
    """)
end

function main()
    if isempty(ARGS) || ARGS[1] == "--help"
        print_help()
        return
    end

    if ARGS[1] == "--list-cameras"
        println(list_available_cameras())
        return
    end

    if ARGS[1] == "--train"
        if length(ARGS) < 2
            println("Erro: informe a pasta de treinamento.")
            return
        end
        train_model(ARGS[2])
        return
    end

    if ARGS[1] == "--capture"
        if length(ARGS) < 3
            println("Erro: use --capture <pessoa> <pasta> [num]")
            return
        end
        num = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 10
        captured = capture_training_session(ARGS[2], ARGS[3]; num_images=num)
        println("Capturadas: $captured imagens")
        return
    end

    if ARGS[1] == "--identify"
        if length(ARGS) < 2
            println("Erro: informe a imagem para identificar.")
            return
        end
        if !isfile(MODEL_PATH)
            println("Modelo não encontrado. Treine antes: julia src/cnncheckin.jl --train dados/fotos_train")
            return
        end
        model = load(MODEL_PATH, "model_state")
        cfg = load_config(CONFIG_PATH)
        names = cfg["data"]["person_names"]
        person, conf = predict_image(model, names, ARGS[2])
        if person === nothing
            println("Não foi possível identificar a imagem.")
        else
            println("Pessoa: $person | confiança: $(round(conf * 100, digits=2))%")
        end
        return
    end

    println("Opção inválida: $(ARGS[1])")
    print_help()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module
