# webcamcnn/config.jl

# Configuration and core utility functions with layer visualization
# VERSÃO CORRIGIDA - Melhor tratamento de dependências

# Importações básicas (sempre necessárias)
using Flux, Images, FileIO, Statistics, Random, JLD2, TOML
using ImageTransformations, LinearAlgebra, Dates, VideoIO, Plots

# Importações opcionais (com fallback)
try
    using ColorSchemes
    global HAS_COLORSCHEMES = true
catch e
    @warn "ColorSchemes não disponível. Usando cores padrão."
    global HAS_COLORSCHEMES = false
end

try
    using CUDA
    global HAS_CUDA = CUDA.functional()
    if HAS_CUDA
        @info "CUDA disponível - GPU será usada para aceleração"
    else
        @info "CUDA não funcional - usando CPU"
    end
catch e
    @warn "CUDA não disponível - usando CPU"
    global HAS_CUDA = false
end

# Global Configuration
const CONFIG = Dict(
    # Model parameters
    :img_size => (128, 128),
    :batch_size => 8,
    :epochs => 30,
    :learning_rate => 0.0001,
    
    # Paths (using absolute paths to avoid issues)
    :data_dir => abspath("../../../dados/webcamcnn"),
    :photos_dir => nothing,  # Will be set in init
    :models_dir => nothing,  # Will be set in init
    :visualizations_dir => nothing, # Will be set in init
    
    # Files
    :model_file => "face_model.jld2",
    :config_file => "system_config.toml",
    :weights_file => "model_weights.toml"
)

# Initialize directories
function init_directories()
    CONFIG[:photos_dir] = joinpath(CONFIG[:data_dir], "photos")
    CONFIG[:models_dir] = joinpath(CONFIG[:data_dir], "models")
    CONFIG[:visualizations_dir] = joinpath(CONFIG[:data_dir], "visualizations")
    
    for dir in [CONFIG[:data_dir], CONFIG[:photos_dir], CONFIG[:models_dir], CONFIG[:visualizations_dir]]
        !isdir(dir) && mkpath(dir)
    end
    
    println("Directories initialized:")
    println("  Data: $(CONFIG[:data_dir])")
    println("  Photos: $(CONFIG[:photos_dir])")  
    println("  Models: $(CONFIG[:models_dir])")
    println("  Visualizations: $(CONFIG[:visualizations_dir])")
end

# Image preprocessing
function preprocess_image(img_path::String; augment=false)
    try
        img = load(img_path)
        
        # Convert to RGB if needed
        if ndims(img) == 2 || eltype(img) <: Gray
            img = RGB.(img)
        elseif eltype(img) <: RGBA
            img = RGB.(img)
        end
        
        # Resize and normalize
        img_resized = imresize(img, CONFIG[:img_size])
        img_array = Float32.(channelview(img_resized))
        img_array = permutedims(img_array, (2, 3, 1))
        
        # Normalize
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
        println("Error processing image $img_path: $e")
        return nothing
    end
end

# Simple data augmentation
function augment_image(img_array::Array{Float32, 3})
    augmented = [img_array]  # Original
    
    # Horizontal flip
    push!(augmented, reverse(img_array, dims=2))
    
    # Brightness variations
    bright = clamp.(img_array .* 1.1, -2.0f0, 2.0f0)
    dark = clamp.(img_array .* 0.9, -2.0f0, 2.0f0)
    push!(augmented, bright, dark)
    
    # Light noise
    noise = img_array .+ 0.02f0 .* randn(Float32, size(img_array))
    push!(augmented, clamp.(noise, -2.0f0, 2.0f0))
    
    return augmented
end

# Extract person name from filename
function extract_person_name(filename::String)
    return split(splitext(filename)[1], "_")[1]
end

# Validate image file
function validate_image_file(filepath::String)
    try
        img = load(filepath)
        return size(img) != (0, 0)
    catch
        return false
    end
end

# Create CNN model
function create_cnn_model(num_classes::Int)
    # Calculate final feature size after max pooling
    final_size = div(CONFIG[:img_size][1], 16)  # 4 max pools = 2^4 = 16
    final_features = 256 * final_size * final_size
    
    model = Chain(
        # Feature extraction
        Conv((3, 3), 3 => 64, relu, pad=1),
        BatchNorm(64),
        MaxPool((2, 2)),
        
        Conv((3, 3), 64 => 128, relu, pad=1), 
        BatchNorm(128),
        MaxPool((2, 2)),
        
        Conv((3, 3), 128 => 256, relu, pad=1),
        BatchNorm(256), 
        MaxPool((2, 2)),
        
        Conv((3, 3), 256 => 256, relu, pad=1),
        BatchNorm(256),
        MaxPool((2, 2)),
        
        # Classification
        Flux.flatten,
        Dense(final_features, 512, relu),
        Dropout(0.4),
        Dense(512, 256, relu), 
        Dropout(0.3),
        Dense(256, num_classes)
    )
    
    # Move to GPU if available
    if HAS_CUDA
        try
            model = model |> gpu
            @info "Model moved to GPU"
        catch e
            @warn "Could not move model to GPU: $e"
        end
    end
    
    return model
end

# Visualize layer activations during processing
function visualize_layer_activations(model, input_image, person_name::String; save_intermediate=true)
    println("Creating layer visualizations for: $person_name")
    
    # Create person-specific visualization directory
    person_viz_dir = joinpath(CONFIG[:visualizations_dir], person_name)
    !isdir(person_viz_dir) && mkpath(person_viz_dir)
    
    # Prepare input
    img_batch = reshape(input_image, size(input_image)..., 1)
    current_activation = img_batch
    
    layer_outputs = []
    layer_names = []
    
    # Process through each layer
    for (i, layer) in enumerate(model)
        try
            current_activation = layer(current_activation)
            push!(layer_outputs, current_activation)
            push!(layer_names, "layer_$(i)_$(typeof(layer).name.name)")
            
            if save_intermediate
                save_layer_visualization(current_activation, i, typeof(layer).name.name, person_viz_dir)
            end
        catch e
            println("Error processing layer $i: $e")
            break
        end
    end
    
    # Create summary visualization
    create_summary_visualization(layer_outputs, layer_names, person_viz_dir, person_name)
    
    return layer_outputs, layer_names
end

# Save individual layer visualization
function save_layer_visualization(activation, layer_idx::Int, layer_type::String, save_dir::String)
    try
        # Handle different activation shapes
        if ndims(activation) >= 4  # Conv layer output (H, W, C, B)
            # Take first batch and create montage of feature maps
            act = activation[:, :, :, 1]  # Remove batch dimension
            num_channels = size(act, 3)
            
            # Create grid of feature maps
            grid_size = ceil(Int, sqrt(num_channels))
            
            # Escolher colorscheme baseado na disponibilidade
            color_scheme = HAS_COLORSCHEMES ? :viridis : :blues
            
            fig = plot(layout=(grid_size, grid_size), size=(800, 800))
            
            for c in 1:min(num_channels, grid_size^2)
                channel_data = act[:, :, c]
                # Normalize for visualization
                if maximum(channel_data) != minimum(channel_data)
                    channel_data = (channel_data .- minimum(channel_data)) ./ (maximum(channel_data) - minimum(channel_data))
                end
                
                heatmap!(fig[c], channel_data, color=color_scheme, aspect_ratio=:equal, 
                        title="Ch $c", showaxis=false, grid=false)
            end
            
            # Hide empty subplots
            for c in (num_channels+1):grid_size^2
                plot!(fig[c], framestyle=:none, grid=false, showaxis=false)
            end
            
            filename = joinpath(save_dir, "layer_$(layer_idx)_$(layer_type)_features.png")
            savefig(fig, filename)
            
        elseif ndims(activation) == 2  # Dense layer output (features, batch)
            # Visualize as bar chart
            act_data = activation[:, 1]  # First batch
            
            color_scheme = HAS_COLORSCHEMES ? :viridis : :blues
            
            fig = bar(1:length(act_data), act_data, 
                     title="Layer $layer_idx - $layer_type Activations",
                     xlabel="Neuron Index", ylabel="Activation Value",
                     color=color_scheme)
            
            filename = joinpath(save_dir, "layer_$(layer_idx)_$(layer_type)_activations.png")
            savefig(fig, filename)
        end
        
    catch e
        println("Warning: Could not visualize layer $layer_idx: $e")
    end
end

# Create comprehensive summary visualization
function create_summary_visualization(layer_outputs, layer_names, save_dir::String, person_name::String)
    try
        # Create summary plots showing the progression through layers
        num_layers = length(layer_outputs)
        fig = plot(layout=(2, 2), size=(1200, 800))
        
        # Plot 1: Activation magnitudes per layer
        avg_activations = []
        max_activations = []
        min_activations = []
        
        for output in layer_outputs
            if ndims(output) >= 2
                flat_output = reshape(output, :)
                push!(avg_activations, mean(abs.(flat_output)))
                push!(max_activations, maximum(flat_output))
                push!(min_activations, minimum(flat_output))
            end
        end
        
        if !isempty(avg_activations)
            plot!(fig[1], 1:length(avg_activations), avg_activations, 
                  title="Average Activation Magnitude", xlabel="Layer", ylabel="Magnitude",
                  marker=:circle, color=:blue, label="Mean |Activation|")
        end
        
        # Plot 2: Layer output shapes
        layer_sizes = []
        for output in layer_outputs
            push!(layer_sizes, prod(size(output)[1:end-1]))  # Exclude batch dimension
        end
        
        if !isempty(layer_sizes)
            bar!(fig[2], 1:length(layer_sizes), layer_sizes,
                 title="Feature Map Sizes", xlabel="Layer", ylabel="Number of Features",
                 color=:green, alpha=0.7)
        end
        
        # Plot 3: Activation distribution for final layer
        if !isempty(layer_outputs)
            final_output = layer_outputs[end]
            if ndims(final_output) >= 2
                final_flat = reshape(final_output, :)
                histogram!(fig[3], final_flat, bins=30, 
                          title="Final Layer Distribution", xlabel="Activation Value", 
                          ylabel="Frequency", color=:red, alpha=0.7)
            end
        end
        
        # Plot 4: Layer type progression
        conv_layers = sum(occursin.("Conv", layer_names))
        dense_layers = sum(occursin.("Dense", layer_names))
        norm_layers = sum(occursin.("BatchNorm", layer_names))
        pool_layers = sum(occursin.("MaxPool", layer_names))
        
        layer_counts = [conv_layers, dense_layers, norm_layers, pool_layers]
        layer_types = ["Conv", "Dense", "BatchNorm", "MaxPool"]
        
        pie!(fig[4], layer_counts, labels=layer_types, title="Layer Type Distribution")
        
        # Save summary
        summary_filename = joinpath(save_dir, "$(person_name)_processing_summary.png")
        savefig(fig, summary_filename)
        
        # Also save a text summary
        text_summary_path = joinpath(save_dir, "$(person_name)_layer_info.txt")
        open(text_summary_path, "w") do io
            println(io, "Layer Processing Summary for: $person_name")
            println(io, "Generated on: $(Dates.now())")
            println(io, "=" ^ 50)
            
            for (i, (output, name)) in enumerate(zip(layer_outputs, layer_names))
                println(io, "Layer $i: $name")
                println(io, "  Shape: $(size(output))")
                if ndims(output) >= 2
                    flat = reshape(output, :)
                    println(io, "  Stats: min=$(round(minimum(flat), digits=4)), max=$(round(maximum(flat), digits=4)), mean=$(round(mean(flat), digits=4))")
                end
                println(io, "")
            end
        end
        
        println("Summary visualizations saved for: $person_name")
        
    catch e
        println("Error creating summary visualization: $e")
    end
end

# Save system configuration
function save_system_config(person_names::Vector{String}, training_info::Dict)
    config_path = joinpath(CONFIG[:models_dir], CONFIG[:config_file])
    
    config_data = Dict(
        "system" => Dict(
            "version" => "4.0-Enhanced-Fixed",
            "created" => string(Dates.now()),
            "img_size" => collect(CONFIG[:img_size]),
            "num_classes" => length(person_names),
            "visualization_enabled" => true,
            "has_cuda" => HAS_CUDA,
            "has_colorschemes" => HAS_COLORSCHEMES
        ),
        "training" => training_info,
        "people" => Dict(
            "names" => person_names,
            "count" => length(person_names)
        )
    )
    
    try
        open(config_path, "w") do io
            TOML.print(io, config_data)
        end
        println("Configuration saved: $config_path")
        return true
    catch e
        println("Error saving config: $e")
        return false
    end
end

# Load system configuration
function load_system_config()
    config_path = joinpath(CONFIG[:models_dir], CONFIG[:config_file])
    
    if !isfile(config_path)
        return nothing
    end
    
    try
        return TOML.parsefile(config_path)
    catch e
        println("Error loading config: $e")
        return nothing
    end
end