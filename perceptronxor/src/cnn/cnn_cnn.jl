# file cnn_cnn.jl
# Script para carregar imagens, normalizá-las, criar batches e treinar uma CNN usando
# basepath = "../../../../dados/imgs"

using Flux
using Flux: flatten
using Images
using FileIO
using Glob
using Random

# Função para carregar imagens e rótulos a partir da pasta base
function load_images_labels(basepath)
    classes = readdir(basepath)
    X = Float32[]
    y = Int[]
    total_imgs = 0
    for (label, class) in enumerate(classes)
        class_path = joinpath(basepath, class)
        pngs = glob("*.png", class_path)
        jpgs = glob("*.jpg", class_path)
        jpegs = glob("*.jpeg", class_path)
        imgfiles = vcat(pngs, jpgs, jpegs)
        println("Classe '$class' tem $(length(imgfiles)) imagens")
        for imgfile in imgfiles
            img = load(imgfile) |> channelview |> float32
            push!(X, img)
            push!(y, label)
            total_imgs += 1
        end
    end
    if total_imgs == 0
        error("Nenhuma imagem carregada. Verifique o caminho e os arquivos da pasta base: $basepath")
    end
    return X, y, classes
end

# Normalização das imagens para [0,1]
function normalize_images(images)
    return [img ./ 255f0 for img in images]
end

# Transformar dados para formato batch (batchsize, canais, altura, largura)
function create_batches(X, y, batchsize)
    n = length(y)
    shuffle = Random.shuffle(1:n)
    batches = []
    labels = []
    for i in 1:batchsize:n - batchsize + 1
        inds = shuffle[i:i + batchsize - 1]
        batchX = cat(X[inds]..., dims=4)  # concatena na dimensão batch
        batchY = Flux.onehotbatch(y[inds], 1:maximum(y))
        push!(batches, batchX)
        push!(labels, batchY)
    end
    return batches, labels
end

# Defina o caminho para a pasta contendo subpastas por classe
basepath = "../../../../dados/imgs"  # ajuste para o caminho correto

# Carregar dados
X, y, classes = load_images_labels(basepath)
X = normalize_images(X)

# Criar batches
batchsize = 16
batches, labels = create_batches(X, y, batchsize)

# Calcular altura e largura após as camadas de pooling para ajustar Dense
altura_inicial = size(X[1], 2)
largura_inicial = size(X[1], 3)
altura_reduzida = altura_inicial ÷ 4  # MaxPool duplo (2x2)
largura_reduzida = largura_inicial ÷ 4

# Definir modelo CNN com flatten do Flux importado
model = Chain(
    Conv((3, 3), 3 => 16, pad = 1, relu),
    MaxPool((2, 2)),
    Conv((3, 3), 16 => 32, pad = 1, relu),
    MaxPool((2, 2)),
    flatten,
    Dense(32 * altura_reduzida * largura_reduzida, 64, relu),
    Dense(64, length(classes)),
    softmax
)

# Função loss e otimizador
loss(x, y) = Flux.crossentropy(model(x), y)
opt = ADAM()

# Treinar modelo por epochs
for epoch in 1:10
    for (x, y) in zip(batches, labels)
        gs = Flux.gradient(() -> loss(x, y), Flux.params(model))
        Flux.Optimise.update!(opt, Flux.params(model), gs)
    end
    println("Epoch $epoch completa")
end
