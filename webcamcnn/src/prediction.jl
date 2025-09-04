# webcamcnn/prediction.jl
# Model prediction and testing functionality with real-time layer visualization

include("config.jl")

# Main prediction interface
function test_model_system()
    println("\nENHANCED MODEL TESTING WITH LAYER VISUALIZATION")
    println("===============================================")
    
    model_path = joinpath(CONFIG[:models_dir], CONFIG[:model_file])
    
    if !isfile(model_path)
        println("No trained model found!")
        println("Please train a model first")
        return false
    end
    
    try
        # Load model
        println("Loading trained model...")
        data = JLD2.load(model_path)
        model = data["model"]
        person_names = data["person_names"]
        
        println("Model loaded successfully!")
        println("Recognized people: $(join(person_names, ", "))")
        
        while true
            println("\n=== TESTING OPTIONS ===")
            println("1 - Live webcam testing with layer visualization")
            println("2 - Test with image file and create visualizations")
            println("3 - Batch test multiple images")
            println("4 - Real-time confidence monitoring")
            println("5 - Compare predictions across different photos")
            println("6 - Analyze model decision process")
            println("7 - Back to main menu")
            
            print("Choose option: ")
            option = strip(readline())
            
            if option == "1"
                live_webcam_testing(model, person_names)
            elseif option == "2"
                test_with_file_visualization(model, person_names)
            elseif option == "3"
                batch_test_images(model, person_names)
            elseif option == "4"
                confidence_monitoring(model, person_names)
            elseif option == "5"
                compare_predictions(model, person_names)
            elseif option == "6"
                analyze_decision_process(model, person_names)
            elseif option == "7"
                return true
            else
                println("Invalid option")
            end
            
            println("\nPress Enter to continue...")
            readline()
        end
        
    catch e
        println("Error loading model: $e")
        return false
    end
end

# Live webcam testing with real-time layer visualization
function live_webcam_testing(model, person_names)
    println("\nLIVE WEBCAM TESTING WITH LAYER VISUALIZATION")
    println("============================================")
    println("Commands:")
    println("  Enter - Capture and analyze")
    println("  'v' + Enter - Capture with full layer visualization")
    println("  'c' + Enter - Continuous prediction mode")
    println("  'q' + Enter - Quit")
    
    try
        camera = VideoIO.opencamera()
        println("Camera started successfully!")
        
        continuous_mode = false
        frame_count = 0
        
        while true
            if continuous_mode
                # Continuous prediction without waiting
                frame = read(camera)
                if frame !== nothing
                    frame_count += 1
                    
                    # Save temporary image
                    temp_path = joinpath(tempdir(), "continuous_$frame_count.jpg")
                    save(temp_path, frame)
                    
                    # Quick prediction
                    prediction, confidence = predict_with_timing(model, person_names, temp_path)
                    
                    if prediction !== nothing
                        conf_bar = "█" ^ Int(round(confidence * 10))
                        println("Frame $frame_count: $prediction ($(round(confidence*100, digits=1))%) $conf_bar")
                    end
                    
                    # Clean up
                    try; rm(temp_path); catch; end
                    
                    # Check for input without blocking
                    sleep(0.1)
                    # Note: In real implementation, you'd want non-blocking input check
                end
                
                # Check if user wants to exit continuous mode
                print("Press 'q' to stop continuous mode: ")
                input = readline()
                if lowercase(strip(input)) == "q"
                    continuous_mode = false
                    println("Exiting continuous mode")
                end
                
            else
                # Manual mode
                print("Command (Enter/v/c/q): ")
                input = strip(readline())
                
                if lowercase(input) == "q"
                    break
                elseif lowercase(input) == "c"
                    continuous_mode = true
                    println("Entering continuous prediction mode...")
                    continue
                end
                
                # Capture frame
                frame = read(camera)
                if frame !== nothing
                    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
                    temp_path = joinpath(tempdir(), "webcam_test_$timestamp.jpg")
                    save(temp_path, frame)
                    
                    if lowercase(input) == "v"
                        # Full visualization mode
                        println("Creating full layer visualization...")
                        prediction, confidence = predict_with_full_visualization(
                            model, person_names, temp_path, "webcam_capture_$timestamp"
                        )
                    else
                        # Standard prediction
                        prediction, confidence = predict_with_timing(model, person_names, temp_path)
                    end
                    
                    if prediction !== nothing
                        println("Prediction: $prediction ($(round(confidence*100, digits=1))% confidence)")
                        
                        # Show confidence bar
                        conf_level = Int(round(confidence * 20))
                        conf_bar = "█" ^ conf_level * "░" ^ (20 - conf_level)
                        println("Confidence: [$conf_bar] $(round(confidence*100, digits=1))%")
                        
                        if confidence < 0.7
                            println("⚠️  Low confidence - consider improving lighting or angle")
                        end
                    else
                        println("Could not make prediction")
                    end
                    
                    # Clean up temp file
                    try; rm(temp_path); catch; end
                else
                    println("Could not capture frame")
                end
            end
        end
        
        close(camera)
        println("Webcam testing completed")
        
    catch e
        println("Error with webcam: $e")
    end
end

# Test with file and create comprehensive visualization
function test_with_file_visualization(model, person_names)
    println("\nFILE TESTING WITH VISUALIZATION")
    println("===============================")
    print("Enter path to image file: ")
    filepath = strip(readline())
    
    if !isfile(filepath)
        println("File not found: $filepath")
        return
    end
    
    println("Testing image: $(basename(filepath))")
    
    # Extract person name from filename for visualization directory
    test_person_name = splitext(basename(filepath))[1]
    
    # Predict with full visualization
    prediction, confidence = predict_with_full_visualization(
        model, person_names, filepath, test_person_name
    )
    
    if prediction !== nothing
        println("\n=== PREDICTION RESULTS ===")
        println("Predicted Person: $prediction")
        println("Confidence: $(round(confidence*100, digits=2))%")
        
        # Show all class probabilities
        show_all_probabilities(model, person_names, filepath)
        
        # Create decision analysis
        create_decision_analysis(model, person_names, filepath, test_person_name)
        
    else
        println("Could not make prediction")
    end
end

# Batch test multiple images
function batch_test_images(model, person_names)
    println("\nBATCH TESTING")
    println("=============")
    print("Enter directory path containing test images: ")
    test_dir = strip(readline())
    
    if !isdir(test_dir)
        println("Directory not found: $test_dir")
        return
    end
    
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    test_images = []
    
    for filename in readdir(test_dir)
        ext = lowercase(splitext(filename)[2])
        if ext in image_extensions
            push!(test_images, joinpath(test_dir, filename))
        end
    end
    
    if isempty(test_images)
        println("No valid images found in directory")
        return
    end
    
    println("Found $(length(test_images)) images to test")
    print("Create visualizations for all? (y/n): ")
    create_viz = lowercase(strip(readline())) in ["y", "yes"]
    
    results = []
    
    for (i, img_path) in enumerate(test_images)
        println("\nTesting image $i/$(length(test_images)): $(basename(img_path))")
        
        if create_viz
            test_name = splitext(basename(img_path))[1]
            prediction, confidence = predict_with_full_visualization(
                model, person_names, img_path, "batch_test_$test_name"
            )
        else
            prediction, confidence = predict_with_timing(model, person_names, img_path)
        end
        
        push!(results, (
            image=basename(img_path),
            prediction=prediction,
            confidence=confidence,
            filepath=img_path
        ))
        
        if prediction !== nothing
            println("Result: $prediction ($(round(confidence*100, digits=1))%)")
        else
            println("Result: No prediction")
        end
    end
    
    # Create batch results summary
    create_batch_results_summary(results, person_names)
end

# Real-time confidence monitoring
function confidence_monitoring(model, person_names)
    println("\nCONFIDENCE MONITORING")
    println("====================")
    println("This will show real-time confidence levels for different people")
    println("Press Enter to capture, 'q' to quit")
    
    try
        camera = VideoIO.opencamera()
        println("Camera started!")
        
        confidence_history = Dict{String, Vector{Float64}}()
        for name in person_names
            confidence_history[name] = Float64[]
        end
        
        capture_count = 0
        
        while true
            input = readline()
            if lowercase(strip(input)) == "q"
                break
            end
            
            frame = read(camera)
            if frame !== nothing
                capture_count += 1
                temp_path = joinpath(tempdir(), "confidence_monitor_$capture_count.jpg")
                save(temp_path, frame)
                
                # Get probabilities for all classes
                probabilities = get_all_class_probabilities(model, person_names, temp_path)
                
                if probabilities !== nothing
                    println("\n--- Capture $capture_count ---")
                    for (i, name) in enumerate(person_names)
                        conf = probabilities[i]
                        push!(confidence_history[name], conf)
                        
                        # Create visual confidence bar
                        bar_length = Int(round(conf * 30))
                        conf_bar = "█" ^ bar_length * "░" ^ (30 - bar_length)
                        
                        println("$name: [$conf_bar] $(round(conf*100, digits=1))%")
                    end
                    
                    # Show trend for most confident prediction
                    best_idx = argmax(probabilities)
                    best_name = person_names[best_idx]
                    println("\nBest match: $best_name")
                    
                    if length(confidence_history[best_name]) >= 3
                        recent_trend = mean(confidence_history[best_name][end-2:end]) - 
                                     mean(confidence_history[best_name][1:min(3, length(confidence_history[best_name]))])
                        trend_arrow = recent_trend > 0.05 ? "↗️" : recent_trend < -0.05 ? "↘️" : "➡️"
                        println("Trend: $trend_arrow")
                    end
                end
                
                try; rm(temp_path); catch; end
            end
        end
        
        close(camera)
        
        # Create confidence history visualization
        create_confidence_history_plot(confidence_history, person_names)
        
    catch e
        println("Error in confidence monitoring: $e")
    end
end

# Compare predictions across different photos
function compare_predictions(model, person_names)
    println("\nPREDICTION COMPARISON")
    println("====================")
    
    photos = []
    while true
        print("Enter image path (or 'done' to finish): ")
        path = strip(readline())
        
        if lowercase(path) == "done"
            break
        end
        
        if isfile(path)
            push!(photos, path)
            println("Added: $(basename(path))")
        else
            println("File not found: $path")
        end
    end
    
    if length(photos) < 2
        println("Need at least 2 photos for comparison")
        return
    end
    
    println("\nComparing $(length(photos)) photos...")
    
    comparison_results = []
    
    for (i, photo_path) in enumerate(photos)
        println("Processing photo $i: $(basename(photo_path))")
        
        probabilities = get_all_class_probabilities(model, person_names, photo_path)
        
        if probabilities !== nothing
            push!(comparison_results, (
                photo=basename(photo_path),
                path=photo_path,
                probabilities=probabilities,
                prediction=person_names[argmax(probabilities)],
                confidence=maximum(probabilities)
            ))
        end
    end
    
    # Create comparison visualization
    create_prediction_comparison_plot(comparison_results, person_names)
    
    # Print comparison table
    println("\n=== COMPARISON RESULTS ===")
    println("Photo Name | Prediction | Confidence | All Probabilities")
    println("-" ^ 70)
    
    for result in comparison_results
        probs_str = join([round(p*100, digits=1) for p in result.probabilities], ", ")
        println("$(result.photo) | $(result.prediction) | $(round(result.confidence*100, digits=1))% | [$probs_str]")
    end
end

# Analyze model decision process
function analyze_decision_process(model, person_names)
    println("\nMODEL DECISION PROCESS ANALYSIS")
    println("===============================")
    
    print("Enter image path for detailed analysis: ")
    filepath = strip(readline())
    
    if !isfile(filepath)
        println("File not found: $filepath")
        return
    end
    
    println("Analyzing decision process for: $(basename(filepath))")
    
    # Create detailed analysis
    analysis_name = "decision_analysis_$(splitext(basename(filepath))[1])"
    create_detailed_decision_analysis(model, person_names, filepath, analysis_name)
    
    println("Decision analysis completed!")
    println("Results saved in visualizations directory")
end

# Core prediction functions

# Predict with timing information
function predict_with_timing(model, person_names, image_path)
    start_time = time()
    
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
        
        end_time = time()
        processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        println("Processing time: $(round(processing_time, digits=1)) ms")
        
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

# Predict with full layer visualization
function predict_with_full_visualization(model, person_names, image_path, viz_name)
    try
        img_arrays = preprocess_image(image_path; augment=false)
        
        if img_arrays === nothing || isempty(img_arrays)
            return nothing, 0.0
        end
        
        # Create visualization
        println("Creating layer visualizations...")
        layer_outputs, layer_names = visualize_layer_activations(model, img_arrays[1], viz_name)
        
        # Make prediction
        img_tensor = reshape(img_arrays[1], size(img_arrays[1])..., 1)
        output = model(img_tensor)
        probabilities = softmax(output)
        
        pred_idx = argmax(probabilities[:, 1])
        confidence = probabilities[pred_idx, 1]
        
        # Save prediction details with visualization
        save_prediction_details(probabilities, person_names, viz_name, image_path)
        
        if pred_idx <= length(person_names)
            return person_names[pred_idx], confidence
        else
            return nothing, 0.0
        end
        
    catch e
        println("Error in prediction with visualization: $e")
        return nothing, 0.0
    end
end

# Get probabilities for all classes
function get_all_class_probabilities(model, person_names, image_path)
    try
        img_arrays = preprocess_image(image_path; augment=false)
        
        if img_arrays === nothing || isempty(img_arrays)
            return nothing
        end
        
        img_tensor = reshape(img_arrays[1], size(img_arrays[1])..., 1)
        output = model(img_tensor)
        probabilities = softmax(output)
        
        return probabilities[:, 1]  # Return probabilities for all classes
        
    catch e
        println("Error getting probabilities: $e")
        return nothing
    end
end

# Show all class probabilities
function show_all_probabilities(model, person_names, image_path)
    probabilities = get_all_class_probabilities(model, person_names, image_path)
    
    if probabilities !== nothing
        println("\n=== ALL CLASS PROBABILITIES ===")
        for (i, name) in enumerate(person_names)
            conf = probabilities[i]
            bar_length = Int(round(conf * 20))
            conf_bar = "█" ^ bar_length * "░" ^ (20 - bar_length)
            
            println("$name: [$conf_bar] $(round(conf*100, digits=2))%")
        end
    end
end

# Save prediction details
function save_prediction_details(probabilities, person_names, viz_name, image_path)
    try
        viz_dir = joinpath(CONFIG[:visualizations_dir], viz_name)
        !isdir(viz_dir) && mkpath(viz_dir)
        
        details_path = joinpath(viz_dir, "prediction_details.txt")
        
        open(details_path, "w") do io
            println(io, "Prediction Details")
            println(io, "==================")
            println(io, "Image: $(basename(image_path))")
            println(io, "Generated: $(Dates.now())")
            println(io, "")
            println(io, "Class Probabilities:")
            
            for (i, name) in enumerate(person_names)
                prob = probabilities[i, 1]
                println(io, "  $name: $(round(prob*100, digits=4))%")
            end
            
            pred_idx = argmax(probabilities[:, 1])
            println(io, "")
            println(io, "Final Prediction: $(person_names[pred_idx])")
            println(io, "Confidence: $(round(probabilities[pred_idx, 1]*100, digits=4))%")
        end
        
    catch e
        println("Error saving prediction details: $e")
    end
end

# Create decision analysis
function create_decision_analysis(model, person_names, image_path, analysis_name)
    try
        analysis_dir = joinpath(CONFIG[:visualizations_dir], analysis_name)
        !isdir(analysis_dir) && mkpath(analysis_dir)
        
        # Get probabilities
        probabilities = get_all_class_probabilities(model, person_names, image_path)
        
        if probabilities !== nothing
            # Create probability distribution plot
            fig = bar(person_names, probabilities .* 100,
                     title="Prediction Probabilities for $(basename(image_path))",
                     xlabel="Person", ylabel="Probability (%)",
                     color=:viridis, alpha=0.7)
            
            # Add confidence threshold line
            hline!(fig, [70], color=:red, linestyle=:dash, label="Confidence Threshold")
            
            savefig(fig, joinpath(analysis_dir, "probability_distribution.png"))
            
            println("Decision analysis saved in: $analysis_dir")
        end
        
    catch e
        println("Error creating decision analysis: $e")
    end
end

# Create detailed decision analysis with layer attention
function create_detailed_decision_analysis(model, person_names, image_path, analysis_name)
    try
        analysis_dir = joinpath(CONFIG[:visualizations_dir], analysis_name)
        !isdir(analysis_dir) && mkpath(analysis_dir)
        
        # Process image through model and capture intermediate outputs
        img_arrays = preprocess_image(image_path; augment=false)
        if img_arrays === nothing || isempty(img_arrays)
            return
        end
        
        # Create layer activations
        layer_outputs, layer_names = visualize_layer_activations(model, img_arrays[1], analysis_name)
        
        # Analyze which layers contribute most to the decision
        create_layer_importance_analysis(layer_outputs, layer_names, analysis_dir)
        
        # Create attention maps for convolutional layers
        create_attention_maps(layer_outputs, layer_names, analysis_dir)
        
        println("Detailed decision analysis completed!")
        
    catch e
        println("Error in detailed decision analysis: $e")
    end
end

# Create layer importance analysis
function create_layer_importance_analysis(layer_outputs, layer_names, analysis_dir)
    try
        layer_magnitudes = []
        layer_variances = []
        
        for (output, name) in zip(layer_outputs, layer_names)
            flat_output = reshape(output, :)
            push!(layer_magnitudes, mean(abs.(flat_output)))
            push!(layer_variances, var(flat_output))
        end
        
        # Plot layer importance metrics
        fig = plot(layout=(2, 1), size=(800, 600))
        
        bar!(fig[1], 1:length(layer_magnitudes), layer_magnitudes,
             title="Layer Activation Magnitudes", xlabel="Layer", ylabel="Mean |Activation|",
             color=:blue, alpha=0.7)
        
        bar!(fig[2], 1:length(layer_variances), layer_variances,
             title="Layer Activation Variances", xlabel="Layer", ylabel="Variance",
             color=:red, alpha=0.7)
        
        savefig(fig, joinpath(analysis_dir, "layer_importance.png"))
        
    catch e
        println("Error creating layer importance analysis: $e")
    end
end

# Create attention maps for convolutional layers
function create_attention_maps(layer_outputs, layer_names, analysis_dir)
    try
        attention_dir = joinpath(analysis_dir, "attention_maps")
        !isdir(attention_dir) && mkpath(attention_dir)
        
        for (i, (output, name)) in enumerate(zip(layer_outputs, layer_names))
            if ndims(output) >= 4  # Convolutional layer
                # Create attention map by averaging across channels
                act = output[:, :, :, 1]  # Remove batch dimension
                attention_map = mean(abs.(act), dims=3)[:, :, 1]  # Average across channels
                
                # Normalize for visualization
                if maximum(attention_map) != minimum(attention_map)
                    attention_map = (attention_map .- minimum(attention_map)) ./ 
                                   (maximum(attention_map) - minimum(attention_map))
                end
                
                fig = heatmap(attention_map, color=:hot, aspect_ratio=:equal,
                             title="Attention Map - $name", showaxis=false)
                
                safe_name = replace(name, r"[^\w]" => "_")
                savefig(fig, joinpath(attention_dir, "attention_$(safe_name).png"))
            end
        end
        
    catch e
        println("Error creating attention maps: $e")
    end
end

# Create batch results summary
function create_batch_results_summary(results, person_names)
    try
        summary_dir = joinpath(CONFIG[:visualizations_dir], "batch_test_summary")
        !isdir(summary_dir) && mkpath(summary_dir)
        
        # Calculate statistics
        successful_predictions = filter(r -> r.prediction !== nothing, results)
        success_rate = length(successful_predictions) / length(results)
        
        if !isempty(successful_predictions)
            avg_confidence = mean([r.confidence for r in successful_predictions])
            
            # Count predictions per person
            prediction_counts = Dict{String, Int}()
            for name in person_names
                prediction_counts[name] = count(r -> r.prediction == name, successful_predictions)
            end
            
            # Create summary plot
            fig = plot(layout=(2, 2), size=(1000, 800))
            
            # Success rate
            pie!(fig[1], [length(successful_predictions), length(results) - length(successful_predictions)],
                 labels=["Successful", "Failed"], title="Prediction Success Rate")
            
            # Confidence distribution
            confidences = [r.confidence for r in successful_predictions]
            histogram!(fig[2], confidences, bins=10, title="Confidence Distribution",
                      xlabel="Confidence", ylabel="Count", color=:blue, alpha=0.7)
            
            # Predictions per person
            bar!(fig[3], collect(keys(prediction_counts)), collect(values(prediction_counts)),
                 title="Predictions per Person", xlabel="Person", ylabel="Count",
                 color=:green, alpha=0.7)
            
            # Confidence vs. prediction accuracy (if ground truth available)
            scatter!(fig[4], confidences, ones(length(confidences)),
                    title="Confidence Levels", xlabel="Confidence", ylabel="Predictions",
                    color=:red, alpha=0.6, markersize=4)
            
            savefig(fig, joinpath(summary_dir, "batch_summary.png"))
            
            # Save text summary
            summary_path = joinpath(summary_dir, "batch_summary.txt")
            open(summary_path, "w") do io
                println(io, "Batch Test Summary")
                println(io, "==================")
                println(io, "Total images tested: $(length(results))")
                println(io, "Successful predictions: $(length(successful_predictions))")
                println(io, "Success rate: $(round(success_rate*100, digits=2))%")
                println(io, "Average confidence: $(round(avg_confidence*100, digits=2))%")
                println(io, "")
                println(io, "Predictions per person:")
                for (name, count) in prediction_counts
                    println(io, "  $name: $count")
                end
            end
            
            println("Batch summary saved in: $summary_dir")
        end
        
    catch e
        println("Error creating batch summary: $e")
    end
end

# Create confidence history plot
function create_confidence_history_plot(confidence_history, person_names)
    try
        history_dir = joinpath(CONFIG[:visualizations_dir], "confidence_history")
        !isdir(history_dir) && mkpath(history_dir)
        
        fig = plot(size=(1000, 600), title="Confidence History Over Time",
                  xlabel="Capture Number", ylabel="Confidence (%)")
        
        for name in person_names
            history = confidence_history[name]
            if !isempty(history)
                plot!(fig, 1:length(history), history .* 100, 
                     label=name, linewidth=2, marker=:circle, markersize=3)
            end
        end
        
        savefig(fig, joinpath(history_dir, "confidence_over_time.png"))
        println("Confidence history plot saved")
        
    catch e
        println("Error creating confidence history plot: $e")
    end
end

# Create prediction comparison plot
function create_prediction_comparison_plot(comparison_results, person_names)
    try
        comparison_dir = joinpath(CONFIG[:visualizations_dir], "prediction_comparison")
        !isdir(comparison_dir) && mkpath(comparison_dir)
        
        # Create heatmap of probabilities
        prob_matrix = hcat([r.probabilities for r in comparison_results]...)
        photo_names = [r.photo for r in comparison_results]
        
        fig = heatmap(prob_matrix, 
                     title="Prediction Probabilities Comparison",
                     xlabel="Photos", ylabel="People",
                     xticks=(1:length(photo_names), photo_names),
                     yticks=(1:length(person_names), person_names),
                     color=:viridis)
        
        savefig(fig, joinpath(comparison_dir, "probability_heatmap.png"))
        println("Prediction comparison plot saved")
        
    catch e
        println("Error creating comparison plot: $e")
    end
end