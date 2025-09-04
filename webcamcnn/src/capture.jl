# webcamcnn/capture.jl
# Photo capture functionality with layer visualization

include("config.jl")

# Main photo capture function
function capture_photos()
    println("PHOTO CAPTURE SYSTEM WITH LAYER VISUALIZATION")
    println("=============================================")
    
    # Get person name
    print("Enter person name: ")
    person_name = strip(readline())
    
    if isempty(person_name)
        println("Error: Name cannot be empty!")
        return false
    end
    
    # Clean person name (remove special characters)
    person_name = replace(person_name, r"[^\w\s-]" => "")
    
    println("\nCapture mode:")
    println("1 - Automatic (10 photos with 3s interval)")
    println("2 - Manual (press Enter for each photo)")
    println("3 - Single photo with immediate layer visualization")
    
    print("Choose mode (1/2/3): ")
    mode = strip(readline())
    
    if mode == "1"
        return capture_automatic(person_name)
    elseif mode == "2" 
        return capture_manual(person_name)
    elseif mode == "3"
        return capture_with_visualization(person_name)
    else
        println("Invalid choice!")
        return false
    end
end

# Automatic capture mode
function capture_automatic(person_name::String)
    num_photos = 10
    interval = 3
    
    println("\nAUTOMATIC CAPTURE MODE")
    println("Person: $person_name")
    println("Photos to capture: $num_photos")
    println("Interval: $interval seconds")
    println("\nInstructions:")
    println("- Position yourself in front of the webcam")
    println("- Change angle for each photo (front, left profile, right profile)")
    println("- Ensure good lighting")
    println("- Press Enter to start")
    
    readline()
    
    return execute_capture(person_name, num_photos, interval, true)
end

# Manual capture mode
function capture_manual(person_name::String)
    println("\nMANUAL CAPTURE MODE") 
    println("Person: $person_name")
    println("Press Enter for each photo, type 'quit' to finish")
    
    return execute_capture(person_name, 0, 0, false)
end

# Capture with real-time layer visualization
function capture_with_visualization(person_name::String)
    println("\nCAPTURE WITH LAYER VISUALIZATION")
    println("Person: $person_name")
    println("This mode will capture a photo and immediately show layer processing")
    println("Press Enter when ready...")
    readline()
    
    try
        camera = VideoIO.opencamera()
        println("Camera started successfully!")
        
        println("Position yourself and press Enter to capture and visualize:")
        readline()
        
        # Capture frame
        frame = read(camera)
        close(camera)
        
        if frame !== nothing
            # Generate filename
            timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
            filename = "$(person_name)_viz_$timestamp.jpg"
            filepath = joinpath(CONFIG[:photos_dir], filename)
            
            # Save image
            save(filepath, frame)
            println("Photo saved: $filename")
            
            # Preprocess for visualization
            img_arrays = preprocess_image(filepath; augment=false)
            
            if img_arrays !== nothing && !isempty(img_arrays)
                # Check if we have a trained model for visualization
                model_path = joinpath(CONFIG[:models_dir], CONFIG[:model_file])
                
                if isfile(model_path)
                    try
                        println("Loading model for layer visualization...")
                        model_data = JLD2.load(model_path)
                        model = model_data["model"]
                        person_names = model_data["person_names"]
                        
                        println("Creating layer visualizations...")
                        layer_outputs, layer_names = visualize_layer_activations(
                            model, img_arrays[1], person_name; save_intermediate=true
                        )
                        
                        println("Layer visualization completed!")
                        println("Visualizations saved in: $(joinpath(CONFIG[:visualizations_dir], person_name))")
                        
                        # Show prediction if possible
                        prediction, confidence = predict_person(model, person_names, filepath)
                        if prediction !== nothing
                            println("Model prediction: $prediction ($(round(confidence*100, digits=1))% confidence)")
                        end
                        
                    catch e
                        println("Could not load model for visualization: $e")
                        println("Photo captured but no layer visualization available")
                    end
                else
                    println("No trained model found for layer visualization")
                    println("Photo captured successfully")
                end
            else
                println("Could not preprocess image for visualization")
            end
            
            return true
        else
            println("Error: Could not capture frame")
            return false
        end
        
    catch e
        println("Error accessing webcam: $e")
        return false
    end
end

# Execute photo capture
function execute_capture(person_name::String, num_photos::Int, interval::Int, automatic::Bool)
    try
        camera = VideoIO.opencamera()
        println("Camera started successfully!")
        
        if automatic
            println("First photo in 5 seconds...")
            sleep(5)
        end
        
        photo_count = 0
        saved_photos = String[]
        
        while automatic ? (photo_count < num_photos) : true
            if !automatic
                println("Position yourself and press Enter (or type 'quit'):")
                input = readline()
                if lowercase(strip(input)) == "quit"
                    break
                end
            end
            
            try
                # Capture frame
                frame = read(camera)
                
                if frame !== nothing
                    # Generate filename
                    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
                    filename = "$(person_name)_$(photo_count + 1)_$timestamp.jpg"
                    filepath = joinpath(CONFIG[:photos_dir], filename)
                    
                    # Save image
                    save(filepath, frame)
                    photo_count += 1
                    push!(saved_photos, filepath)
                    
                    println("Photo $photo_count saved: $filename")
                    
                    if automatic && photo_count < num_photos
                        println("Next photo in $interval seconds... Change angle!")
                        sleep(interval)
                    end
                else
                    println("Error: Could not capture frame")
                    if automatic
                        break
                    end
                end
                
            catch e
                println("Error during capture: $e")
                if automatic
                    break
                end
            end
        end
        
        close(camera)
        
        if photo_count > 0
            println("\nCapture completed successfully!")
            println("$photo_count photos saved in: $(CONFIG[:photos_dir])")
            println("\nCaptured photos:")
            for (i, photo) in enumerate(saved_photos)
                println("   $i. $(basename(photo))")
            end
            
            # Ask if user wants to create layer visualizations for all photos
            if photo_count > 1
                model_path = joinpath(CONFIG[:models_dir], CONFIG[:model_file])
                if isfile(model_path)
                    print("Create layer visualizations for all captured photos? (y/n): ")
                    response = strip(lowercase(readline()))
                    if response in ["y", "yes", "s", "sim"]
                        create_batch_visualizations(saved_photos, person_name)
                    end
                end
            end
            
            return true
        else
            println("No photos were captured.")
            return false
        end
        
    catch e
        println("Error accessing webcam: $e")
        println("\nTroubleshooting tips:")
        println("- Check if webcam is connected")
        println("- Close other programs using the webcam")
        println("- Run with appropriate permissions")
        println("- Ensure webcam drivers are installed")
        return false
    end
end

# Create layer visualizations for multiple photos
function create_batch_visualizations(photo_paths::Vector{String}, person_name::String)
    println("Creating batch layer visualizations...")
    
    model_path = joinpath(CONFIG[:models_dir], CONFIG[:model_file])
    
    try
        model_data = JLD2.load(model_path)
        model = model_data["model"]
        person_names = model_data["person_names"]
        
        for (i, photo_path) in enumerate(photo_paths)
            println("Processing photo $i/$(length(photo_paths)): $(basename(photo_path))")
            
            img_arrays = preprocess_image(photo_path; augment=false)
            if img_arrays !== nothing && !isempty(img_arrays)
                # Create subdirectory for this specific photo
                photo_name = splitext(basename(photo_path))[1]
                photo_viz_dir = joinpath(CONFIG[:visualizations_dir], person_name, photo_name)
                !isdir(photo_viz_dir) && mkpath(photo_viz_dir)
                
                # Save layer visualizations in photo-specific directory
                layer_outputs, layer_names = visualize_layer_activations_to_dir(
                    model, img_arrays[1], photo_viz_dir, photo_name
                )
                
                # Create prediction info
                prediction, confidence = predict_person(model, person_names, photo_path)
                if prediction !== nothing
                    # Save prediction info
                    pred_info_path = joinpath(photo_viz_dir, "prediction_info.txt")
                    open(pred_info_path, "w") do io
                        println(io, "Prediction Results for: $photo_name")
                        println(io, "Generated on: $(Dates.now())")
                        println(io, "=" ^ 40)
                        println(io, "Predicted Person: $prediction")
                        println(io, "Confidence: $(round(confidence*100, digits=2))%")
                        println(io, "Model Classes: $(join(person_names, ", "))")
                    end
                end
            end
        end
        
        println("Batch visualization completed!")
        println("All visualizations saved in: $(joinpath(CONFIG[:visualizations_dir], person_name))")
        
    catch e
        println("Error creating batch visualizations: $e")
    end
end

# Visualize layers to specific directory
function visualize_layer_activations_to_dir(model, input_image, save_dir::String, photo_name::String)
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
            
            save_layer_visualization_to_dir(current_activation, i, typeof(layer).name.name, save_dir)
        catch e
            println("Error processing layer $i: $e")
            break
        end
    end
    
    # Create summary visualization
    create_summary_visualization_to_dir(layer_outputs, layer_names, save_dir, photo_name)
    
    return layer_outputs, layer_names
end

# Save layer visualization to specific directory
function save_layer_visualization_to_dir(activation, layer_idx::Int, layer_type::String, save_dir::String)
    try
        if ndims(activation) >= 4  # Conv layer output
            act = activation[:, :, :, 1]
            num_channels = size(act, 3)
            
            grid_size = ceil(Int, sqrt(num_channels))
            fig = plot(layout=(grid_size, grid_size), size=(800, 800))
            
            for c in 1:min(num_channels, grid_size^2)
                channel_data = act[:, :, c]
                if maximum(channel_data) != minimum(channel_data)
                    channel_data = (channel_data .- minimum(channel_data)) ./ (maximum(channel_data) - minimum(channel_data))
                end
                
                heatmap!(fig[c], channel_data, color=:viridis, aspect_ratio=:equal, 
                        title="Ch $c", showaxis=false, grid=false)
            end
            
            for c in (num_channels+1):grid_size^2
                plot!(fig[c], framestyle=:none, grid=false, showaxis=false)
            end
            
            filename = joinpath(save_dir, "layer_$(layer_idx)_$(layer_type)_features.png")
            savefig(fig, filename)
            
        elseif ndims(activation) == 2  # Dense layer output
            act_data = activation[:, 1]
            
            fig = bar(1:length(act_data), act_data, 
                     title="Layer $layer_idx - $layer_type Activations",
                     xlabel="Neuron Index", ylabel="Activation Value",
                     color=:viridis)
            
            filename = joinpath(save_dir, "layer_$(layer_idx)_$(layer_type)_activations.png")
            savefig(fig, filename)
        end
        
    catch e
        println("Warning: Could not visualize layer $layer_idx: $e")
    end
end

# Create summary visualization in specific directory
function create_summary_visualization_to_dir(layer_outputs, layer_names, save_dir::String, photo_name::String)
    try
        num_layers = length(layer_outputs)
        fig = plot(layout=(2, 2), size=(1200, 800))
        
        # Same logic as in config.jl but save to specific directory
        avg_activations = []
        for output in layer_outputs
            if ndims(output) >= 2
                flat_output = reshape(output, :)
                push!(avg_activations, mean(abs.(flat_output)))
            end
        end
        
        if !isempty(avg_activations)
            plot!(fig[1], 1:length(avg_activations), avg_activations, 
                  title="Average Activation Magnitude", xlabel="Layer", ylabel="Magnitude",
                  marker=:circle, color=:blue, label="Mean |Activation|")
        end
        
        layer_sizes = [prod(size(output)[1:end-1]) for output in layer_outputs]
        if !isempty(layer_sizes)
            bar!(fig[2], 1:length(layer_sizes), layer_sizes,
                 title="Feature Map Sizes", xlabel="Layer", ylabel="Number of Features",
                 color=:green, alpha=0.7)
        end
        
        if !isempty(layer_outputs)
            final_output = layer_outputs[end]
            if ndims(final_output) >= 2
                final_flat = reshape(final_output, :)
                histogram!(fig[3], final_flat, bins=30, 
                          title="Final Layer Distribution", xlabel="Activation Value", 
                          ylabel="Frequency", color=:red, alpha=0.7)
            end
        end
        
        # Layer type counts
        conv_layers = sum(occursin.("Conv", layer_names))
        dense_layers = sum(occursin.("Dense", layer_names))
        norm_layers = sum(occursin.("BatchNorm", layer_names))
        pool_layers = sum(occursin.("MaxPool", layer_names))
        
        layer_counts = [conv_layers, dense_layers, norm_layers, pool_layers]
        layer_types = ["Conv", "Dense", "BatchNorm", "MaxPool"]
        
        pie!(fig[4], layer_counts, labels=layer_types, title="Layer Type Distribution")
        
        summary_filename = joinpath(save_dir, "$(photo_name)_processing_summary.png")
        savefig(fig, summary_filename)
        
    catch e
        println("Error creating summary visualization: $e")
    end
end

# Simple prediction function for capture module
function predict_person(model, person_names, image_path)
    try
        img_arrays = preprocess_image(image_path; augment=false)
        
        if img_arrays === nothing || isempty(img_arrays)
            return nothing, 0.0
        end
        
        img_tensor = reshape(img_arrays[1], size(img_arrays[1])..., 1)
        output = model(img_tensor)
        probabilities = softmax(output)
        
        pred_idx = argmax(probabilities[:, 1])
        confidence = probabilities[pred_idx, 1]
        
        if pred_idx <= length(person_names)
            return person_names[pred_idx], confidence
        else
            return nothing, 0.0
        end
        
    catch e
        println("Error making prediction: $e")
        return nothing, 0.0
    end
end

# Check if we have sufficient training data
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

# List available people in dataset
function list_people()
    if !isdir(CONFIG[:photos_dir])
        println("No photos directory found")
        return String[]
    end
    
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    people_count = Dict{String, Int}()
    
    for filename in readdir(CONFIG[:photos_dir])
        ext = lowercase(splitext(filename)[2])
        if ext in image_extensions
            filepath = joinpath(CONFIG[:photos_dir], filename)
            if validate_image_file(filepath)
                person = extract_person_name(filename)
                people_count[person] = get(people_count, person, 0) + 1
            end
        end
    end
    
    if isempty(people_count)
        println("No valid photos found")
        return String[]
    end
    
    println("People in dataset:")
    for (person, count) in sort(collect(people_count))
        println("  - $person: $count photos")
    end
    
    return collect(keys(people_count))
end