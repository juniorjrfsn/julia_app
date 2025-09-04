# projeto: webcamcnn
# file: webcamcnn/src/layer_visualization.jl

using Flux
using Images
using FileIO
using ImageTransformations
using Statistics
using Dates

include("core.jl")
using .CNNCheckinCore

# Constantes para visualização
const VIZ_OUTPUT_PATH = "../../../dados/webcamcnn/layer_visualizations"
const MAX_FILTERS_PER_LAYER = 16  # Máximo de filtros para visualizar por layer
const VIZ_IMG_SIZE = (64, 64)     # Tamanho das imagens de visualização

# Estrutura para armazenar ativações das camadas
struct LayerActivation
    layer_name::String
    layer_index::Int
    activations::Array{Float32}
    input_image_name::String
    person_name::String
end

# Função para criar diretórios de visualização organizados por pessoa
function create_visualization_directories(base_path::String, person_names::Vector{String})
    println("📁 Criando estrutura de diretórios para visualizações...")
    
    # Criar diretório base
    if !isdir(base_path)
        mkpath(base_path)
    end
    
    # Criar diretórios por pessoa
    for person_name in person_names
        person_dir = joinpath(base_path, person_name)
        if !isdir(person_dir)
            mkpath(person_dir)
        end
        
        # Criar subdiretórios por layer (assumindo até 10 layers principais)
        for layer_idx in 1:10
            layer_dir = joinpath(person_dir, "layer_$(layer_idx)")
            if !isdir(layer_dir)
                mkpath(layer_dir)
            end
        end
    end
    
    println("✅ Estrutura de diretórios criada em: $base_path")
end

# Função para extrair ativações de uma camada específica
function extract_layer_activations(model, input_batch, layer_indices::Vector{Int})
    activations = Dict{Int, Array{Float32}}()
    
    # Executar forward pass até cada camada de interesse
    x = input_batch
    for (i, layer) in enumerate(model)
        x = layer(x)
        
        if i in layer_indices
            # Converter para Array se necessário
            activation = x isa AbstractArray ? Array(x) : x
            activations[i] = Float32.(activation)
        end
    end
    
    return activations
end

# Função para normalizar ativações para visualização
function normalize_activation_for_viz(activation::Array{Float32})
    # Normalizar entre 0 e 1
    min_val = minimum(activation)
    max_val = maximum(activation)
    
    if max_val > min_val
        normalized = (activation .- min_val) ./ (max_val - min_val)
    else
        normalized = zeros(Float32, size(activation))
    end
    
    return normalized
end

# Função para converter ativação em imagem visualizável
function activation_to_image(activation::Array{Float32}, target_size::Tuple{Int, Int} = VIZ_IMG_SIZE)
    if ndims(activation) == 4
        # Para ativações convolucionais (batch, height, width, channels)
        # Pegar primeiro item do batch e primeira/média dos canais
        if size(activation, 4) > 1
            # Média dos canais
            img_data = mean(activation[1, :, :, :], dims=3)[:, :, 1]
        else
            img_data = activation[1, :, :, 1]
        end
    elseif ndims(activation) == 3
        # Para ativações (height, width, channels)
        if size(activation, 3) > 1
            img_data = mean(activation, dims=3)[:, :, 1]
        else
            img_data = activation[:, :, 1]
        end
    elseif ndims(activation) == 2
        # Para ativações 2D
        img_data = activation
    else
        # Para ativações 1D (Dense layers), criar uma representação visual
        vec_size = length(activation)
        side_size = Int(ceil(sqrt(vec_size)))
        padded = zeros(Float32, side_size^2)
        padded[1:vec_size] = activation[:]
        img_data = reshape(padded, side_size, side_size)
    end
    
    # Normalizar
    img_data = normalize_activation_for_viz(img_data)
    
    # Redimensionar se necessário
    if size(img_data) != target_size
        img_data = imresize(img_data, target_size)
    end
    
    # Converter para formato de imagem
    return Gray.(img_data)
end

# Função para salvar visualização de filtros convolucionais
function save_conv_filter_visualizations(layer, layer_idx::Int, person_name::String, 
                                       image_name::String, base_path::String)
    if !isa(layer, Conv) || !hasfield(typeof(layer), :weight) || layer.weight === nothing
        return false
    end
    
    try
        weights = layer.weight
        # Weights shape: (filter_height, filter_width, input_channels, output_channels)
        
        num_filters = min(size(weights, 4), MAX_FILTERS_PER_LAYER)
        
        for filter_idx in 1:num_filters
            # Extrair filtro específico
            filter_weights = weights[:, :, :, filter_idx]
            
            # Se múltiplos canais de entrada, fazer média
            if size(filter_weights, 3) > 1
                filter_2d = mean(filter_weights, dims=3)[:, :, 1]
            else
                filter_2d = filter_weights[:, :, 1]
            end
            
            # Normalizar e converter para imagem
            filter_normalized = normalize_activation_for_viz(Float32.(filter_2d))
            filter_img = Gray.(filter_normalized)
            
            # Redimensionar para melhor visualização
            if size(filter_img) != (32, 32)
                filter_img = imresize(filter_img, (32, 32))
            end
            
            # Salvar
            filename = "$(image_name)_layer$(layer_idx)_filter$(filter_idx).png"
            save_path = joinpath(base_path, person_name, "layer_$(layer_idx)", filename)
            save(save_path, filter_img)
        end
        
        return true
    catch e
        println("⚠️ Erro ao salvar filtros da camada $layer_idx: $e")
        return false
    end
end

# Função para salvar ativações de uma camada
function save_layer_activations(activations::Dict{Int, Array{Float32}}, 
                               person_name::String, image_name::String, 
                               base_path::String)
    saved_count = 0
    
    for (layer_idx, activation) in activations
        try
            # Converter ativação para imagem
            viz_img = activation_to_image(activation)
            
            # Salvar ativação
            filename = "$(image_name)_layer$(layer_idx)_activation.png"
            save_path = joinpath(base_path, person_name, "layer_$(layer_idx)", filename)
            save(save_path, viz_img)
            
            saved_count += 1
        catch e
            println("⚠️ Erro ao salvar ativação da camada $layer_idx: $e")
        end
    end
    
    return saved_count
end

# Função principal para gerar visualizações durante o treinamento
function generate_training_visualizations(model, train_data, person_names::Vector{String})
    println("🎨 Gerando visualizações das camadas durante treinamento...")
    
    # Criar estrutura de diretórios
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    viz_base_path = joinpath(VIZ_OUTPUT_PATH, "training_$timestamp")
    create_visualization_directories(viz_base_path, person_names)
    
    # Definir camadas de interesse (pular Dropout e outras camadas sem parâmetros)
    layer_indices = []
    for (i, layer) in enumerate(model)
        if isa(layer, Conv) || isa(layer, Dense) || isa(layer, BatchNorm)
            push!(layer_indices, i)
        end
    end
    
    println("🎯 Camadas selecionadas para visualização: $layer_indices")
    
    # Processar algumas imagens de cada pessoa (máximo 3 por pessoa)
    images_per_person = 3
    total_visualizations = 0
    
    # Organizar dados por pessoa
    person_images = Dict{String, Vector{Tuple}}()
    for (batch_images, batch_labels) in train_data
        for i in 1:size(batch_images, 4)
            img = batch_images[:, :, :, i:i]  # Manter dimensões do batch
            label_idx = Flux.onecold(batch_labels[:, i])
            person_name = person_names[label_idx]
            
            if !haskey(person_images, person_name)
                person_images[person_name] = []
            end
            
            if length(person_images[person_name]) < images_per_person
                push!(person_images[person_name], (img, i))
            end
        end
    end
    
    # Gerar visualizações para cada pessoa
    for (person_name, images) in person_images
        println("👤 Processando visualizações para: $person_name")
        
        for (img_idx, (img, original_idx)) in enumerate(images)
            image_name = "sample_$(img_idx)"
            
            # Extrair ativações das camadas
            activations = extract_layer_activations(model, img, layer_indices)
            
            # Salvar ativações
            saved_activations = save_layer_activations(activations, person_name, 
                                                     image_name, viz_base_path)
            
            # Salvar filtros das camadas convolucionais
            for layer_idx in layer_indices
                if layer_idx <= length(model)
                    save_conv_filter_visualizations(model[layer_idx], layer_idx, 
                                                  person_name, image_name, viz_base_path)
                end
            end
            
            total_visualizations += saved_activations
        end
    end
    
    # Criar arquivo de índice
    create_visualization_index(viz_base_path, person_names, layer_indices)
    
    println("✅ Visualizações geradas: $total_visualizations ativações")
    println("📁 Salvas em: $viz_base_path")
    
    return viz_base_path
end

# Função para criar um índice das visualizações
function create_visualization_index(viz_path::String, person_names::Vector{String}, 
                                   layer_indices::Vector{Int})
    index_content = """
# Índice de Visualizações de Camadas
Gerado em: $(Dates.format(now(), "dd/mm/yyyy HH:MM:SS"))

## Estrutura dos Diretórios:
```
$(basename(viz_path))/
$(join(["├── $person/\n$(join(["│   ├── layer_$idx/" for idx in layer_indices], "\n"))" for person in person_names], "\n"))
```

## Pessoas Incluídas:
$(join(["- $person" for person in person_names], "\n"))

## Camadas Visualizadas:
$(join(["- Layer $idx" for idx in layer_indices], "\n"))

## Tipos de Arquivos:
- `*_activation.png`: Ativações da camada para a imagem
- `*_filter*.png`: Filtros convolucionais da camada

## Como Interpretar:
1. **Ativações**: Mostram como a rede "vê" a imagem em cada camada
2. **Filtros**: Mostram quais padrões cada filtro detecta
3. **Camadas iniciais**: Detectam bordas e texturas básicas
4. **Camadas finais**: Detectam características mais complexas e específicas
"""
    
    index_path = joinpath(viz_path, "README.md")
    open(index_path, "w") do f
        write(f, index_content)
    end
    
    println("📄 Índice criado: $index_path")
end

# Função para comparar visualizações entre treinamentos
function compare_layer_visualizations(viz_path1::String, viz_path2::String, 
                                    output_path::String = "comparison_report.md")
    println("🔍 Comparando visualizações entre treinamentos...")
    
    # Esta função seria mais complexa na implementação real
    # Por agora, criar um relatório básico
    
    comparison_report = """
# Relatório de Comparação de Visualizações

## Treinamento 1: $(basename(viz_path1))
## Treinamento 2: $(basename(viz_path2))

Data da comparação: $(Dates.format(now(), "dd/mm/yyyy HH:MM:SS"))

## Análise:
- Compare as ativações das mesmas camadas entre os dois treinamentos
- Observe como os filtros evoluem entre treinamentos
- Identifique padrões que emergem com mais dados/épocas

## Observações:
1. Camadas iniciais tendem a ser mais estáveis
2. Camadas finais mostram maior variação com diferentes dados
3. Filtros mortos (todos zeros) indicam possíveis problemas

## Próximos Passos:
- Analise visualmente os arquivos em ambos os diretórios
- Compare a qualidade das características extraídas
- Ajuste hiperparâmetros se necessário
"""
    
    open(output_path, "w") do f
        write(f, comparison_report)
    end
    
    println("📊 Relatório de comparação salvo em: $output_path")
    return output_path
end

# Integração com o sistema de treinamento existente
function add_visualization_to_training(model, train_data, person_names::Vector{String}, 
                                     epoch::Int, save_frequency::Int = 5)
    # Salvar visualizações apenas em épocas específicas
    if epoch % save_frequency == 0 || epoch == 1
        println("🎨 Salvando visualizações da época $epoch...")
        viz_path = generate_training_visualizations(model, train_data, person_names)
        return viz_path
    end
    return nothing
end

# Export das funções principais
export generate_training_visualizations, create_visualization_directories,
       add_visualization_to_training, compare_layer_visualizations,
       VIZ_OUTPUT_PATH