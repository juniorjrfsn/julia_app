 # module cnncheckin 
module cnncheckin

# projeto: cnncheckin
# file: cnncheckin/src/cnncheckin.jl
# Sistema de Reconhecimento Facial para Check-in
# Autor: Assistente IA
# Descrição: Treina uma CNN para reconhecimento de faces

using Flux
using Images
using FileIO
using CUDA
using Statistics
using Random
using JLD2
using ImageTransformations
using LinearAlgebra
using Dates

# Configurações globais - AJUSTADAS PARA REDUZIR OVERFITTING
const IMG_SIZE = (128, 128)  # Aumentado para melhor qualidade
const BATCH_SIZE = 8  # Reduzido para datasets pequenos
const EPOCHS = 50  # Reduzido para evitar overfitting
const LEARNING_RATE = 0.0001  # Reduzido para aprendizado mais suave
const DATA_PATH = "../../../dados/fotos"
const MODEL_PATH = "face_recognition_model.jld2"

# Estrutura para armazenar dados de uma pessoa
struct PersonData
    name::String
    images::Vector{Array{Float32, 3}}
    label::Int
end

# Função para aplicar data augmentation
function augment_image(img_array::Array{Float32, 3})
    augmented = []
    push!(augmented, img_array)  # Original
    
    # Flip horizontal
    flipped = reverse(img_array, dims=2)
    push!(augmented, flipped)
    
    # Ajuste de brilho
    bright = clamp.(img_array .* 1.2, 0.0f0, 1.0f0)
    dark = clamp.(img_array .* 0.8, 0.0f0, 1.0f0)
    push!(augmented, bright)
    push!(augmented, dark)
    
    # Adicionar ruído
    noise = img_array .+ 0.05f0 .* randn(Float32, size(img_array))
    noise_clamped = clamp.(noise, -2.0f0, 2.0f0)  # Para z-score normalization
    push!(augmented, noise_clamped)
    
    return augmented
end

# Função para carregar e preprocessar uma imagem - MELHORADA
function preprocess_image(img_path::String; augment::Bool = false)
    try
        # Carregar imagem
        img = load(img_path)
        
        # Converter para RGB se necessário
        if ndims(img) == 2
            img = Gray.(img)
            img = RGB.(img)
        elseif isa(img, Array) && eltype(img) <: RGBA
            # RGBA -> RGB
            img = RGB.(img)
        end
        
        # Redimensionar para tamanho padrão
        img_resized = imresize(img, IMG_SIZE)
        
        # Converter para array Float32
        img_array = Float32.(channelview(img_resized))
        
        # Reordenar dimensões para (height, width, channels)
        img_array = permutedims(img_array, (2, 3, 1))
        
        # Normalização mais simples (0-1) primeiro, depois Z-score
        img_array = Float32.(img_array)
        
        # Normalização Z-score apenas se não for constante
        μ = mean(img_array)
        σ = std(img_array)
        if σ > 1e-6
            img_array = (img_array .- μ) ./ σ
        end
        
        if augment
            return augment_image(img_array)
        else
            return [img_array]
        end
    catch e
        println("Erro ao processar imagem $img_path: $e")
        return nothing
    end
end

# Função para extrair nome da pessoa do nome do arquivo
function extract_person_name(filename::String)
    # Remove extensão e pega parte antes do traço
    name_parts = split(splitext(filename)[1], "-")
    return name_parts[1]
end

# Função para carregar dados das imagens - MELHORADA COM DATA AUGMENTATION
function load_face_data(data_path::String; use_augmentation::Bool = true)
    println("📄 Carregando dados das imagens...")
    
    if !isdir(data_path)
        error("Diretório $data_path não encontrado!")
    end
    
    # Dicionário para armazenar imagens por pessoa
    person_images = Dict{String, Vector{Array{Float32, 3}}}()
    
    # Listar todos os arquivos de imagem
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    
    for filename in readdir(data_path)
        file_ext = lowercase(splitext(filename)[2])
        
        if file_ext in image_extensions
            person_name = extract_person_name(filename)
            img_path = joinpath(data_path, filename)
            
            # Preprocessar imagem com ou sem augmentation
            img_arrays = preprocess_image(img_path; augment=use_augmentation)
            
            if img_arrays !== nothing
                if !haskey(person_images, person_name)
                    person_images[person_name] = Vector{Array{Float32, 3}}()
                end
                
                for img_array in img_arrays
                    push!(person_images[person_name], img_array)
                end
                
                total_imgs = use_augmentation ? length(img_arrays) : 1
                println("✅ Carregada: $filename -> $person_name ($total_imgs variações)")
            else
                println("⚠️  Falha ao carregar: $filename")
            end
        end
    end
    
    # Converter para formato adequado
    people_data = Vector{PersonData}()
    person_names = sort(collect(keys(person_images)))
    
    for (idx, person_name) in enumerate(person_names)
        images = person_images[person_name]
        if length(images) > 0
            push!(people_data, PersonData(person_name, images, idx))
            println("👤 Pessoa: $person_name - $(length(images)) imagens")
        end
    end
    
    return people_data, person_names
end

# Função para criar datasets balanceados - MELHORADA
function create_datasets(people_data::Vector{PersonData}, split_ratio::Float64 = 0.8)
    println("📊 Criando datasets de treino e validação...")
    
    train_images = Vector{Array{Float32, 3}}()
    train_labels = Vector{Int}()
    val_images = Vector{Array{Float32, 3}}()
    val_labels = Vector{Int}()
    
    # Dividir por pessoa para manter balanceamento
    for person in people_data
        n_imgs = length(person.images)
        n_train = max(1, Int(floor(n_imgs * split_ratio)))
        
        # Embaralhar imagens desta pessoa
        indices = randperm(n_imgs)
        
        # Treino
        for i in 1:n_train
            push!(train_images, person.images[indices[i]])
            push!(train_labels, person.label)
        end
        
        # Validação
        for i in (n_train+1):n_imgs
            push!(val_images, person.images[indices[i]])
            push!(val_labels, person.label)
        end
    end
    
    println("📈 Dataset criado:")
    println("   - Treino: $(length(train_images)) imagens")
    println("   - Validação: $(length(val_images)) imagens")
    
    return (train_images, train_labels), (val_images, val_labels)
end

# Função para criar batches
function create_batches(images, labels, batch_size)
    batches = []
    n_samples = length(images)
    
    if n_samples == 0
        return batches
    end
    
    # Determinar o número máximo de classes para one-hot encoding
    max_label = maximum(labels)
    
    for i in 1:batch_size:n_samples
        end_idx = min(i + batch_size - 1, n_samples)
        batch_images = images[i:end_idx]
        batch_labels = labels[i:end_idx]
        
        # Converter para tensor 4D (height, width, channels, batch_size)
        batch_tensor = cat(batch_images..., dims=4)
        
        # One-hot encoding dos labels
        batch_labels_onehot = Flux.onehotbatch(batch_labels, 1:max_label)
        
        push!(batches, (batch_tensor, batch_labels_onehot))
    end
    
    return batches
end

# Arquitetura da CNN - MELHORADA CONTRA OVERFITTING
function create_cnn_model(num_classes::Int, input_size::Tuple{Int, Int} = IMG_SIZE)
    # Calcular o tamanho após as convoluções e pooling
    # 128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8
    final_size = div(div(div(div(input_size[1], 2), 2), 2), 2)  # 8
    final_features = 128 * final_size * final_size  # 128 * 8 * 8 = 8192
    
    return Chain(
        # Primeira camada convolucional - mais conservadora
        Conv((3, 3), 3 => 32, relu, pad=1),
        BatchNorm(32),
        Dropout(0.1),  # Dropout leve nas convoluções
        MaxPool((2, 2)),
        
        # Segunda camada convolucional
        Conv((3, 3), 32 => 64, relu, pad=1),
        BatchNorm(64),
        Dropout(0.1),
        MaxPool((2, 2)),
        
        # Terceira camada convolucional
        Conv((3, 3), 64 => 128, relu, pad=1),
        BatchNorm(128),
        Dropout(0.2),
        MaxPool((2, 2)),
        
        # Quarta camada convolucional
        Conv((3, 3), 128 => 128, relu, pad=1),  # Não aumenta canais
        BatchNorm(128),
        Dropout(0.2),
        MaxPool((2, 2)),
        
        # Flatten e camadas densas - MUITO mais conservadoras
        Flux.flatten,
        Dense(final_features, 256, relu),  # Reduzido drasticamente
        Dropout(0.5),
        Dense(256, 64, relu),  # Camada intermediária menor
        Dropout(0.4),
        Dense(64, num_classes)  # Sem softmax aqui, aplicado na loss
    )
end

# Função para calcular acurácia
function accuracy(model, data_loader)
    correct = 0
    total = 0
    
    for (x, y) in data_loader
        ŷ = softmax(model(x))  # Aplicar softmax para predição
        pred = Flux.onecold(ŷ)
        true_labels = Flux.onecold(y)
        correct += sum(pred .== true_labels)
        total += length(true_labels)
    end
    
    return total > 0 ? correct / total : 0.0
end

# Função de treinamento - MELHORADA COM EARLY STOPPING
function train_model(model, train_data, val_data, epochs, learning_rate)
    println("🚀 Iniciando treinamento...")
    
    # Configurar otimizador com weight decay
    optimizer = ADAM(learning_rate, (0.9, 0.999), 1e-8)  # Parâmetros mais conservadores
    opt_state = Flux.setup(optimizer, model)
    
    # Histórico de treinamento
    train_losses = Float64[]
    val_accuracies = Float64[]
    best_val_acc = 0.0
    patience_counter = 0
    patience_limit = 10  # Early stopping
    
    # Loop de treinamento
    for epoch in 1:epochs
        epoch_loss = 0.0
        num_batches = 0
        
        # Treinar por um epoch
        for (x, y) in train_data
            loss, grads = Flux.withgradient(model) do m
                ŷ = m(x)
                # Cross entropy com logits (mais estável numericamente)
                Flux.logitcrossentropy(ŷ, y)
            end
            
            # Atualizar parâmetros
            Flux.update!(opt_state, model, grads[1])
            
            epoch_loss += loss
            num_batches += 1
        end
        
        # Calcular métricas
        avg_loss = num_batches > 0 ? epoch_loss / num_batches : 0.0
        val_acc = accuracy(model, val_data)
        
        push!(train_losses, avg_loss)
        push!(val_accuracies, val_acc)
        
        # Early stopping logic
        if val_acc > best_val_acc
            best_val_acc = val_acc
            patience_counter = 0
        else
            patience_counter += 1
        end
        
        # Exibir progresso
        if epoch % 5 == 0 || epoch == 1
            println("Epoch $epoch/$epochs - Loss: $(round(avg_loss, digits=4)) - Val Acc: $(round(val_acc*100, digits=2))% - Best: $(round(best_val_acc*100, digits=2))%")
        end
        
        # Parar se não melhorar
        if patience_counter >= patience_limit
            println("🛑 Early stopping - sem melhoria por $patience_limit epochs")
            break
        end
        
        # Parar se accuracy boa o suficiente (mas não perfeita)
        if val_acc >= 0.85 && val_acc < 0.98  # Evita overfitting
            println("🎯 Acurácia boa alcançada sem overfitting!")
            break
        end
    end
    
    return train_losses, val_accuracies
end

# Função para salvar modelo
function save_model(model, person_names, filepath::String)
    println("💾 Salvando modelo treinado...")
    
    # Criar dicionário com informações do modelo
    model_data = Dict(
        "model_state" => model,
        "person_names" => person_names,
        "img_size" => IMG_SIZE,
        "num_classes" => length(person_names),
        "timestamp" => string(Dates.now())
    )
    
    # Salvar usando JLD2
    jldsave(filepath; model_data=model_data)
    println("✅ Modelo salvo em: $filepath")
end

# Função para carregar modelo salvo
function load_model(filepath::String)
    println("📂 Carregando modelo salvo...")
    
    if !isfile(filepath)
        error("Arquivo do modelo não encontrado: $filepath")
    end
    
    data = load(filepath)
    model_data = data["model_data"]
    
    return model_data["model_state"], model_data["person_names"]
end

# Função para fazer predição em uma nova imagem - MELHORADA
function predict_person(model, person_names, img_path::String)
    # Preprocessar imagem (sem augmentation para predição)
    img_arrays = preprocess_image(img_path; augment=false)
    
    if img_arrays === nothing || length(img_arrays) == 0
        return nothing, 0.0
    end
    
    img_array = img_arrays[1]  # Pegar apenas a original
    
    # Adicionar dimensão de batch
    img_tensor = reshape(img_array, size(img_array)..., 1)
    
    # Fazer predição
    logits = model(img_tensor)
    prediction = softmax(logits)  # Aplicar softmax
    
    # Obter classe e confiança
    pred_probs = vec(prediction)
    pred_class = argmax(pred_probs)
    confidence = pred_probs[pred_class]
    
    person_name = person_names[pred_class]
    
    return person_name, confidence
end

# Função de treinamento (comando --treino)
function train_command()
    println("🎯 Sistema de Reconhecimento Facial - Modo Treinamento")
    
    try
        # 1. Carregar dados com data augmentation
        people_data, person_names = load_face_data(DATA_PATH; use_augmentation=true)
        
        if length(people_data) == 0
            error("Nenhum dado encontrado!")
        end
        
        num_classes = length(person_names)
        println("👥 Total de pessoas: $num_classes")
        
        # Verificar se há dados suficientes
        total_images = sum(length(person.images) for person in people_data)
        println("📊 Total de imagens (com augmentation): $total_images")
        
        # 2. Criar datasets balanceados
        (train_images, train_labels), (val_images, val_labels) = create_datasets(people_data)
        
        # 3. Criar batches
        train_batches = create_batches(train_images, train_labels, BATCH_SIZE)
        val_batches = create_batches(val_images, val_labels, BATCH_SIZE)
        
        if length(train_batches) == 0
            error("Não foi possível criar batches de treinamento!")
        end
        
        # 4. Criar modelo
        println("🧠 Criando modelo CNN...")
        model = create_cnn_model(num_classes)
        
        # 5. Treinar modelo
        train_losses, val_accuracies = train_model(model, train_batches, val_batches, EPOCHS, LEARNING_RATE)
        
        # 6. Exibir resultados finais
        final_acc = length(val_accuracies) > 0 ? val_accuracies[end] : 0.0
        println("🎉 Treinamento concluído!")
        println("📊 Acurácia final: $(round(final_acc*100, digits=2))%")
        
        # 7. Salvar modelo
        save_model(model, person_names, MODEL_PATH)
        
        println("✅ Sistema treinado e salvo com sucesso!")
        
        return true
        
    catch e
        println("❌ Erro durante treinamento: $e")
        return false
    end
end

# Função de identificação (comando --identifica)
function identify_command(image_path::String)
    println("🔍 Sistema de Reconhecimento Facial - Modo Identificação")
    println("📸 Analisando imagem: $image_path")
    
    try
        # Verificar se o arquivo existe
        if !isfile(image_path)
            error("Arquivo de imagem não encontrado: $image_path")
        end
        
        # Verificar se o modelo existe
        if !isfile(MODEL_PATH)
            error("Modelo não encontrado! Execute primeiro: julia cnncheckin.jl --treino")
        end
        
        # Carregar modelo
        model, person_names = load_model(MODEL_PATH)
        
        # Fazer predição
        person_name, confidence = predict_person(model, person_names, image_path)
        
        if person_name === nothing
            println("❌ Não foi possível processar a imagem")
            return false
        end
        
        # Exibir resultado
        println("🎯 Resultado da Identificação:")
        println("   👤 Pessoa: $person_name")
        println("   📈 Confiança: $(round(confidence*100, digits=2))%")
        
        # Avaliar confiança de forma mais conservadora
        if confidence >= 0.7
            println("✅ Identificação com alta confiança")
        elseif confidence >= 0.5
            println("⚠️  Identificação com confiança média")
        else
            println("❓ Identificação com baixa confiança - pode ser pessoa desconhecida")
        end
        
        return true
        
    catch e
        println("❌ Erro durante identificação: $e")
        return false
    end
end

# Função para exibir ajuda
function show_help()
    println("🎯 Sistema de Reconhecimento Facial CNN Check-in")
    println()
    println("Uso:")
    println("  julia cnncheckin.jl --treino                    # Treinar o modelo")
    println("  julia cnncheckin.jl --identifica <imagem>       # Identificar pessoa")
    println("  julia cnncheckin.jl --help                      # Mostrar esta ajuda")
    println()
    println("Exemplos:")
    println("  julia cnncheckin.jl --treino")
    println("  julia cnncheckin.jl --identifica ../dados/teste.jpg")
    println("  julia cnncheckin.jl --identifica photo.jpeg")
    println()
    println("Estrutura de diretórios esperada:")
    println("  $DATA_PATH/")
    println("    ├── pessoa1-001.jpg")
    println("    ├── pessoa1-002.jpg")
    println("    ├── pessoa2-001.jpg")
    println("    └── ...")
    println()
    println("Formatos suportados: .jpg, .jpeg, .png, .bmp, .tiff")
    println()
    println("🔧 Configurações atuais:")
    println("  - Tamanho da imagem: $(IMG_SIZE)")
    println("  - Batch size: $BATCH_SIZE")
    println("  - Epochs máximos: $EPOCHS")
    println("  - Learning rate: $LEARNING_RATE")
end

# Função principal para parsing de argumentos
function parse_args_and_run()
    if length(ARGS) == 0
        show_help()
        return
    end
    
    command = ARGS[1]
    
    if command == "--treino"
        println("🚀 Iniciando modo treinamento...")
        success = train_command()
        if success
            println("🎉 Treinamento concluído com sucesso!")
        else
            println("💥 Falha no treinamento")
        end
        
    elseif command == "--identifica"
        if length(ARGS) < 2
            println("❌ Erro: Caminho da imagem não fornecido")
            println("Uso: julia cnncheckin.jl --identifica <caminho_da_imagem>")
            return
        end
        
        image_path = ARGS[2]
        success = identify_command(image_path)
        if success
            println("✅ Identificação concluída!")
        else
            println("💥 Falha na identificação")
        end
        
    elseif command == "--help" || command == "-h"
        show_help()
        
    else
        println("❌ Comando não reconhecido: $command")
        println("Use --help para ver os comandos disponíveis")
        show_help()
    end
end

# Função principal original (para compatibilidade)
function main()
    println("⚠️  Aviso: Use os novos comandos CLI para melhor experiência!")
    println("Execute: julia cnncheckin.jl --help")
    
    return train_command()
end

# Função de saudação original
greet() = print("Hello World!")

# Exportar funções principais
export main, predict_person, load_model, save_model, greet, train_command, identify_command

end # module cnncheckin

# Executar o parser de argumentos se for o arquivo principal
if abspath(PROGRAM_FILE) == @__FILE__
    cnncheckin.parse_args_and_run()
end

#  julia cnncheckin.jl --treino

#  julia cnncheckin.jl --identifica ../../../dados/fotos_teste/534770020_18019526477744454_2931624826193581596_n.jpg