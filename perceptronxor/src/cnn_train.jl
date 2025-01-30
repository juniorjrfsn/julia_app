using Flux, Zygote, Images, ImageIO, BSON, ProgressMeter, Random, MLUtils, Statistics
import Flux.Optimise: Adam
using Flux: onehotbatch

function carregar_dados(caminho_diretorio)
    if !isdir(caminho_diretorio)
        error("O diretório '$caminho_diretorio' não existe.")
    end

    X = Vector{Array{Float32, 3}}()  # Array de imagens 3D (canais × altura × largura)
    y = Vector{Array{Float32, 2}}()  # Array de rótulos one-hot
    classes = readdir(caminho_diretorio)
    p = Progress(length(classes), desc="Carregando dados: ")

    for (classe_id, classe_nome) in enumerate(classes)
        caminho_classe = joinpath(caminho_diretorio, classe_nome)
        for arquivo_imagem in readdir(caminho_classe)
            if match(r".*\.(jpg|jpeg|png|gif)$", arquivo_imagem) !== nothing
                caminho_imagem = joinpath(caminho_classe, arquivo_imagem)
                try
                    img = load(caminho_imagem)

                    # Verificar se a imagem já tem 3 canais
                    if ndims(img) == 3 && size(img, 3) == 3
                        img_redimensionada = imresize(img, (64, 64))
                    elseif ndims(img) == 2  # Imagem em escala de cinza
                        img_rgb = reinterpret(RGB{Float32}, repeat(img, Inner(3)))
                        img_redimensionada = imresize(img_rgb, (64, 64))
                    else
                        println("Imagem $caminho_imagem não é RGB de 3 canais ou grayscale, ignorando.")
                        continue
                    end

                    # Verificar se a imagem redimensionada tem 3 canais
                    if ndims(img_redimensionada) != 3 || size(img_redimensionada, 3) != 3
                        println("A imagem $caminho_imagem não possui 3 canais RGB após redimensionamento. Dimensões: $(size(img_redimensionada)), ignorando.")
                        continue
                    end

                    # Reordenar as dimensões para (canais × altura × largura)
                    img_redimensionada = permutedims(channelview(img_redimensionada), (3, 1, 2))
                    push!(X, Float32.(img_redimensionada))
                    push!(y, Float32.(onehotbatch(classe_id, 1:length(classes))))

                catch e
                    println("Erro ao carregar imagem $caminho_imagem: $e")
                end
            end
        end
        next!(p)
    end

    if isempty(X) || isempty(y)
        error("Não foram carregadas imagens válidas ou rótulos.")
    end

    X_array = cat(X..., dims=4)
    y_array = cat(y..., dims=2)

    if !all(isfinite, X_array)
        error("Há elementos não numéricos ou não finitos em X_array. Verifique as imagens carregadas.")
    end

    X_mean = mean(X_array, dims=(1, 2, 3))
    X_std = std(X_array, dims=(1, 2, 3))
    if any(X_std .== 0)
        error("Desvio padrão zero detectado. Verifique as imagens carregadas.")
    end
    X_array = (X_array .- X_mean) ./ (X_std .+ 1e-6)

    n_treino = Int(floor(0.8 * size(X_array, 4)))
    X_treino, y_treino = X_array[:, :, :, 1:n_treino], y_array[:, 1:n_treino]
    X_val, y_val = X_array[:, :, :, n_treino+1:end], y_array[:, n_treino+1:end]

    return X_treino, y_treino, X_val, y_val
end

# Carregar os dados
X_treino, y_treino, X_val, y_val = carregar_dados("img_train")

# Definir o modelo CNN
modelo = Chain(
    Conv((3, 3), 3 => 16, relu),
    BatchNorm(16),
    MaxPool((2, 2)),
    Dropout(0.2),
    Conv((3, 3), 16 => 32, relu),
    BatchNorm(32),
    MaxPool((2, 2)),
    Conv((3, 3), 32 => 64, relu),
    BatchNorm(64),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(64 * 8 * 8, 10),
    softmax
)

# Definir a função de perda com regularização L2
perda(x, y) = Flux.Losses.crossentropy(modelo(x), y) + 0.001 * sum(abs2, Flux.params(modelo))
otimizador = Adam()

# Função de treinamento com validação e mini-batch gradient descent
function treinar_modelo!(modelo, X_treino, y_treino, X_val, y_val, otimizador, epocas; batch_size=32)
    n = size(X_treino, 4)
    batches = Flux.minibatches((X_treino, y_treino), batch_size)

    for epoca in 1:epocas
        Flux.train!(perda, Flux.params(modelo), batches, otimizador)

        treino_loss = sum(perda(X_treino[:, :, :, i], y_treino[:, i]) for i in 1:n) / n
        val_loss = sum(perda(X_val[:, :, :, i], y_val[:, i]) for i in 1:size(X_val, 4)) / size(X_val, 4)

        println("Época: $epoca, Perda de Treinamento: $treino_loss, Perda de Validação: $val_loss")
    end
end

# Treinar o modelo
treinar_modelo!(modelo, X_treino, y_treino, X_val, y_val, otimizador, 10)

# Salvar o modelo treinado
BSON.@save "modelo_treinado.bson" modelo



## Execute ##
# $ cd .\perceptronxor\src\
# $ julia cnn_train.jl
