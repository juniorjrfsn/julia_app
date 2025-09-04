# webcamcnn/training.jl
# Model training functionality with layer analysis and weight visualization

include("config.jl")

# Main training function with enhanced layer analysis
function train_model()
    println("\nENHANCED MODEL TRAINING SYSTEM WITH LAYER ANALYSIS")
    println("==================================================")
    
    # Check if we have training data
    data_ok, msg = check_training_data()
    if !data_ok
        println("Error: $msg")
        println("Please capture photos first")
        return false
    end
    
    println("Data status: $msg")
    
    try
        # Load and prepare data
        println("\nLoading training data...")
        people_data, person_names = load_training_data()
        
        if isempty(people_data)
            println("Error: No valid training data found")
            return false
        end
        
        num_classes = length(person_names)
        println("People: $num_classes")
        println("Names: $(join(person_names, ", "))")
        
        # Create datasets
        train_data, val_data = create_datasets(people_data)
        println("Training samples: $(length(train_data))")
        println("Validation samples: $(length(val_data))")
        
        # Create and train model
        println("\nCreating CNN model...")
        model = create_cnn_model(num_classes)
        
        # Analyze model architecture before training
        analyze_model_architecture(model, person_names)
        
        println("\nStarting training with layer monitoring...")
        training_info = train_cnn_model_with_analysis(model, train_data, val_data, person_names)
        
        if training_info["success"]
            # Save model and configuration
            model_path = joinpath(CONFIG[:models_dir], CONFIG[:model_file])
            JLD2.save(model_path, "model", model, "person_names", person_names)
            
            # Save configuration
            save_system_config(person_names, training_info)
            
            # Save weights to TOML with enhanced metadata
            save_enhanced_weights_toml(model, person_names, training_info)
            
            # Create training visualizations for each person
            create_training_visualizations(model, people_data, person_names)
            
            println("\nTraining completed successfully!")
            println("Best accuracy: $(round(training_info["best_accuracy"]*100, digits=2))%")
            println("Model saved: $model_path")
            println("Layer visualizations created for all people")
            
            return true
        else
            println("Training failed!")
            return false
        end
        
    catch e
        println("Error during training: $e")
        return false
    end
end

# Load training data from photos directory
function load_training_data()
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    person_images = Dict{String, Vector{Array{Float32, 3}}}()
    
    for filename in readdir(CONFIG[:photos_dir])
        ext = lowercase(splitext(filename)[2])
        if ext in image_extensions
            img_path = joinpath(CONFIG[:photos_dir], filename)
            
            if !validate_image_file(img_path)
                continue
            end
            
            person_name = extract_person_name(filename)
            img_arrays = preprocess_image(img_path; augment=true)
            
            if img_arrays !== nothing
                if !haskey(person_images, person_name)
                    person_images[person_name] = Vector{Array{Float32, 3}}()
                end
                
                for img_array in img_arrays
                    push!(person_images[person_name], img_array)
                end
                
                println("Loaded: $filename -> $person_name ($(length(img_arrays)) variations)")
            end
        end
    end
    
    # Convert to structured format
    people_data = []
    person_names = sort(collect(keys(person_images)))
    
    for (idx, person_name) in enumerate(person_names)
        images = person_images[person_name]
        if !isempty(images)
            push!(people_data, (name=person_name, images=images, label=idx))
            println("Person: $person_name - $(length(images)) images (Label: $idx)")
        end
    end
    
    return people_data, person_names
end

# Create training and validation datasets
function create_datasets(people_data; split_ratio=0.8)
    train_images = []
    train_labels = []
    val_images = []
    val_labels = []
    
    for person in people_data
        n_imgs = length(person.images)
        n_train = max(1, Int(floor(n_imgs * split_ratio)))
        indices = randperm(n_imgs)
        
        # Training set
        for i in 1:n_train
            push!(train_images, person.images[indices[i]])
            push!(train_labels, person.label)
        end
        
        # Validation set
        for i in (n_train+1):n_imgs
            push!(val_images, person.images[indices[i]])
            push!(val_labels, person.label)
        end
        
        println("$(person.name): $n_train training, $(n_imgs - n_train) validation")
    end
    
    return (train_images, train_labels), (val_images, val_labels)
end

# Analyze model architecture
function analyze_model_architecture(model, person_names::Vector{String})
    println("\nMODEL ARCHITECTURE ANALYSIS")
    println("===========================")
    
    total_params = 0
    conv_params = 0
    dense_params = 0
    
    for (i, layer) in enumerate(model)
        layer_params = 0
        layer_type = typeof(layer).name.name
        
        if hasfield(typeof(layer), :weight) && layer.weight !== nothing
            layer_params += length(layer.weight)
        end
        if hasfield(typeof(layer), :bias) && layer.bias !== nothing
            layer_params += length(layer.bias)
        end
        
        total_params += layer_params
        
        if isa(layer, Conv)
            conv_params += layer_params
        elseif isa(layer, Dense)
            dense_params += layer_params
        end
        
        println("Layer $i ($layer_type): $layer_params parameters")
    end
    
    println("\nModel Summary:")
    println("- Total parameters: $total_params")
    println("- Convolutional parameters: $conv_params ($(round(conv_params/total_params*100, digits=1))%)")
    println("- Dense parameters: $dense_params ($(round(dense_params/total_params*100, digits=1))%)")
    println("- Model size: $(round(total_params * 4 / (1024^2), digits=2)) MB")
    println("- Output classes: $(length(person_names))")
end

# Train CNN model with enhanced analysis
function train_cnn_model_with_analysis(model, train_data, val_data, person_names)
    # Create batches
    train_batches = create_batches(train_data[1], train_data[2], CONFIG[:batch_size])
    val_batches = create_batches(val_data[1], val_data[2], CONFIG[:batch_size])
    
    if isempty(train_batches)
        return Dict("success" => false, "error" => "No training batches created")
    end
    
    # Setup optimizer
    optimizer = ADAM(CONFIG[:learning_rate])
    opt_state = Flux.setup(optimizer, model)
    
    # Training variables
    best_accuracy = 0.0
    best_epoch = 0
    patience_counter = 0
    patience_limit = 10
    
    # Storage for training analysis
    epoch_losses = Float64[]
    epoch_accuracies = Float64[]
    layer_stats_history = []
    
    start_time = time()
    
    for epoch in 1:CONFIG[:epochs]
        println("\nEpoch $epoch/$(CONFIG[:epochs])")
        
        # Training phase
        epoch_loss = 0.0
        num_batches = 0
        
        for (x, y) in train_batches
            try
                loss, grads = Flux.withgradient(model) do m
                    ŷ = m(x)
                    Flux.logitcrossentropy(ŷ, y)
                end
                
                Flux.update!(opt_state, model, grads[1])
                epoch_loss += loss
                num_batches += 1
            catch e
                println("Error in training batch: $e")
                continue
            end
        end
        
        avg_loss = num_batches > 0 ? epoch_loss / num_batches : Inf
        push!(epoch_losses, avg_loss)
        
        # Validation phase
        val_acc = calculate_accuracy(model, val_batches)
        push!(epoch_accuracies, val_acc)
        
        # Analyze layer statistics
        layer_stats = analyze_layer_statistics(model)
        push!(layer_stats_history, layer_stats)
        
        println("Loss: $(round(avg_loss, digits=6)) - Val Acc: $(round(val_acc*100, digits=2))%")
        
        # Print layer statistics periodically
        if epoch % 5 == 0
            print_layer_statistics_summary(layer_stats, epoch)
        end
        
        # Early stopping check
        if val_acc > best_accuracy
            best_accuracy = val_acc
            best_epoch = epoch
            patience_counter = 0
            println("New best accuracy: $(round(best_accuracy*100, digits=2))%")
            
            # Save layer visualizations at best epoch
            if epoch > 5  # Skip early epochs
                create_best_epoch_visualizations(model, train_data, person_names, epoch)
            end
        else
            patience_counter += 1
            if patience_counter >= patience_limit
                println("Early stopping at epoch $epoch")
                break
            end
        end
        
        # Progress indicator
        progress = epoch / CONFIG[:epochs] * 100
        println("Progress: $(round(progress, digits=1))% - Best epoch: $best_epoch")
    end
    
    end_time = time()
    duration = (end_time - start_time) / 60
    
    # Create training analysis visualization
    create_training_analysis_plots(epoch_losses, epoch_accuracies, layer_stats_history, person_names)
    
    return Dict(
        "success" => true,
        "best_accuracy" => best_accuracy,
        "best_epoch" => best_epoch,
        "epochs_trained" => min(best_epoch + patience_limit, CONFIG[:epochs]),
        "duration_minutes" => duration,
        "person_names" => person_names,
        "model_architecture" => "CNN_Face_Recognition_Enhanced",
        "learning_rate" => CONFIG[:learning_rate],
        "batch_size" => CONFIG[:batch_size],
        "final_loss" => epoch_losses[end],
        "training_losses" => epoch_losses,
        "validation_accuracies" => epoch_accuracies
    )
end

# Create batches for training
function create_batches(images, labels, batch_size)
    batches = []
    n_samples = length(images)
    
    if n_samples == 0
        return batches
    end
    
    unique_labels = unique(labels)
    label_range = minimum(unique_labels):maximum(unique_labels)
    
    for i in 1:batch_size:n_samples
        end_idx = min(i + batch_size - 1, n_samples)
        batch_images = images[i:end_idx]
        batch_labels = labels[i:end_idx]
        
        batch_tensor = cat(batch_images..., dims=4)
        
        try
            batch_labels_onehot = Flux.onehotbatch(batch_labels, label_range)
            push!(batches, (batch_tensor, batch_labels_onehot))
        catch e
            println("Error creating batch $i-$end_idx: $e")
            continue
        end
    end
    
    return batches
end

# Calculate model accuracy
function calculate_accuracy(model, data_batches)
    correct = 0
    total = 0
    
    for (x, y) in data_batches
        try
            ŷ = softmax(model(x))
            pred = Flux.onecold(ŷ)
            true_labels = Flux.onecold(y)
            correct += sum(pred .== true_labels)
            total += length(true_labels)
        catch e
            println("Error calculating accuracy: $e")
            continue
        end
    end
    
    return total > 0 ? correct / total : 0.0
end

# Analyze layer statistics during training
function analyze_layer_statistics(model)
    layer_stats = Dict{String, Dict{String, Float64}}()
    
    for (i, layer) in enumerate(model)
        layer_name = "layer_$(i)_$(typeof(layer).name.name)"
        stats = Dict{String, Float64}()
        
        if hasfield(typeof(layer), :weight) && layer.weight !== nothing
            weights = layer.weight
            stats["weight_mean"] = mean(weights)
            stats["weight_std"] = std(weights)
            stats["weight_min"] = minimum(weights)
            stats["weight_max"] = maximum(weights)
            stats["weight_norm"] = norm(weights)
        end
        
        if hasfield(typeof(layer), :bias) && layer.bias !== nothing
            bias = layer.bias
            stats["bias_mean"] = mean(bias)
            stats["bias_std"] = std(bias)
            stats["bias_norm"] = norm(bias)
        end
        
        layer_stats[layer_name] = stats
    end
    
    return layer_stats
end

# Print layer statistics summary
function print_layer_statistics_summary(layer_stats, epoch)
    println("Layer Statistics (Epoch $epoch):")
    for (layer_name, stats) in layer_stats
        if haskey(stats, "weight_norm")
            println("  $layer_name: W_norm=$(round(stats["weight_norm"], digits=4))")
        end
    end
end

# Create training analysis plots
function create_training_analysis_plots(losses, accuracies, layer_stats_history, person_names)
    try
        # Create training analysis directory
        analysis_dir = joinpath(CONFIG[:visualizations_dir], "training_analysis")
        !isdir(analysis_dir) && mkpath(analysis_dir)
        
        # Plot training curves
        fig1 = plot(layout=(2, 1), size=(800, 600))
        
        plot!(fig1[1], 1:length(losses), losses, 
              title="Training Loss", xlabel="Epoch", ylabel="Loss",
              color=:red, linewidth=2)
        
        plot!(fig1[2], 1:length(accuracies), accuracies .* 100,
              title="Validation Accuracy", xlabel="Epoch", ylabel="Accuracy (%)",
              color=:blue, linewidth=2)
        
        savefig(fig1, joinpath(analysis_dir, "training_curves.png"))
        
        # Plot layer weight evolution
        if !isempty(layer_stats_history)
            create_weight_evolution_plots(layer_stats_history, analysis_dir)
        end
        
        println("Training analysis plots saved in: $analysis_dir")
        
    catch e
        println("Error creating training analysis plots: $e")
    end
end

# Create weight evolution plots
function create_weight_evolution_plots(layer_stats_history, analysis_dir)
    try
        # Extract layer names
        layer_names = collect(keys(layer_stats_history[1]))
        
        for layer_name in layer_names
            weight_norms = []
            bias_norms = []
            
            for stats_dict in layer_stats_history
                if haskey(stats_dict, layer_name)
                    layer_stats = stats_dict[layer_name]
                    if haskey(layer_stats, "weight_norm")
                        push!(weight_norms, layer_stats["weight_norm"])
                    end
                    if haskey(layer_stats, "bias_norm")
                        push!(bias_norms, layer_stats["bias_norm"])
                    end
                end
            end
            
            if !isempty(weight_norms)
                fig = plot(1:length(weight_norms), weight_norms,
                          title="$layer_name Weight Evolution",
                          xlabel="Epoch", ylabel="Weight Norm",
                          color=:green, linewidth=2)
                
                safe_layer_name = replace(layer_name, r"[^\w]" => "_")
                savefig(fig, joinpath(analysis_dir, "$(safe_layer_name)_weights.png"))
            end
        end
        
    catch e
        println("Error creating weight evolution plots: $e")
    end
end

# Create best epoch visualizations
function create_best_epoch_visualizations(model, train_data, person_names, epoch)
    try
        best_epoch_dir = joinpath(CONFIG[:visualizations_dir], "best_epoch_$epoch")
        !isdir(best_epoch_dir) && mkpath(best_epoch_dir)
        
        # Create visualizations for one sample from each person
        for person_name in person_names
            # Find first sample for this person
            for (img, label) in zip(train_data[1], train_data[2])
                if label == findfirst(==(person_name), person_names)
                    visualize_layer_activations_to_dir(model, img, best_epoch_dir, person_name)
                    break
                end
            end
        end
        
        println("Best epoch visualizations saved in: $best_epoch_dir")
        
    catch e
        println("Error creating best epoch visualizations: $e")
    end
end

# Create training visualizations for all people
function create_training_visualizations(model, people_data, person_names)
    try
        training_viz_dir = joinpath(CONFIG[:visualizations_dir], "final_training_results")
        !isdir(training_viz_dir) && mkpath(training_viz_dir)
        
        for person in people_data
            person_name = person.name
            println("Creating final visualizations for: $person_name")
            
            # Use first image for visualization
            if !isempty(person.images)
                img = person.images[1]
                person_dir = joinpath(training_viz_dir, person_name)
                !isdir(person_dir) && mkpath(person_dir)
                
                visualize_layer_activations_to_dir(model, img, person_dir, "$(person_name)_final")
            end
        end
        
        println("Final training visualizations completed!")
        
    catch e
        println("Error creating training visualizations: $e")
    end
end

# Enhanced weights saving
function save_enhanced_weights_toml(model, person_names::Vector{String}, training_info::Dict)
    try
        weights_file = joinpath(CONFIG[:models_dir], "enhanced_model_weights.toml")
        
        # Create enhanced training metadata
        training_id = "train_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
        
        weights_data = Dict(
            "training_id" => training_id,
            "timestamp" => string(Dates.now()),
            "model_info" => Dict(
                "architecture" => "CNN_Enhanced",
                "classes" => person_names,
                "num_classes" => length(person_names),
                "input_size" => collect(CONFIG[:img_size])
            ),
            "training_results" => Dict(
                "best_accuracy" => training_info["best_accuracy"],
                "best_epoch" => training_info["best_epoch"],
                "final_loss" => training_info["final_loss"],
                "epochs_trained" => training_info["epochs_trained"],
                "duration_minutes" => training_info["duration_minutes"]
            ),
            "layer_analysis" => analyze_final_model_state(model)
        )
        
        open(weights_file, "w") do io
            TOML.print(io, weights_data)
        end
        
        println("Enhanced weights metadata saved: $weights_file")
        return true
        
    catch e
        println("Error saving enhanced weights: $e")
        return false
    end
end

# Analyze final model state
function analyze_final_model_state(model)
    analysis = Dict{String, Any}()
    
    for (i, layer) in enumerate(model)
        layer_name = "layer_$(i)_$(typeof(layer).name.name)"
        layer_analysis = Dict{String, Any}()
        
        if hasfield(typeof(layer), :weight) && layer.weight !== nothing
            w = layer.weight
            layer_analysis["weight_stats"] = Dict(
                "mean" => mean(w),
                "std" => std(w),
                "min" => minimum(w),
                "max" => maximum(w),
                "shape" => collect(size(w))
            )
        end
        
        if hasfield(typeof(layer), :bias) && layer.bias !== nothing
            b = layer.bias
            layer_analysis["bias_stats"] = Dict(
                "mean" => mean(b),
                "std" => std(b),
                "shape" => collect(size(b))
            )
        end
        
        analysis[layer_name] = layer_analysis
    end
    
    return analysis
end

# Check training data (reused from original)
function check_training_data()
    if !isdir(CONFIG[:photos_dir])
        return false, "Photos directory does not exist"
    end
    
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    valid_images = 0
    people = Set{String}()
    
    for filename in readdir(CONFIG[:photos_dir])
        ext = lowercase(splitext(filename)[2])
        if ext in image_extensions
            filepath = joinpath(CONFIG[:photos_dir], filename)
            if validate_image_file(filepath)
                valid_images += 1
                person = extract_person_name(filename)
                push!(people, person)
            end
        end
    end
    
    num_people = length(people)
    
    if num_people < 1
        return false, "No people found in data"
    end
    
    if valid_images < 5
        return false, "Too few valid images ($valid_images). Minimum recommended: 5"
    end
    
    return true, "Valid data: $num_people person(s), $valid_images image(s)"
end