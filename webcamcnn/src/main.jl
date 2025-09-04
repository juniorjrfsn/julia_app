#!/usr/bin/env julia
# webcamcnn/main.jl
# Main system interface - Enhanced version with layer visualization


include("config.jl")
include("capture.jl")
include("training.jl")
include("prediction.jl")

const SYSTEM_VERSION = "4.0-Enhanced-LayerViz"

function print_header()
    println("🧠 ENHANCED CNN FACE RECOGNITION SYSTEM v$SYSTEM_VERSION")
    println("🎯 With Real-time Layer Visualization & Deep Analysis")
    println("=" ^ 65)
    println("Date: $(Dates.format(now(), "dd/mm/yyyy HH:MM:SS"))")
    println()
end

function print_system_status()
    println("📊 SYSTEM STATUS:")
    println("-" ^ 25)
    
    # Check directories
    dirs = [
        (CONFIG[:data_dir], "📁 Data directory"),
        (CONFIG[:photos_dir], "📸 Photos directory"),  
        (CONFIG[:models_dir], "🤖 Models directory"),
        (CONFIG[:visualizations_dir], "🎨 Visualizations directory")
    ]
    
    for (dir, desc) in dirs
        status = isdir(dir) ? "✅ OK" : "❌ Missing"
        println("$desc: $status")
        if isdir(dir)
            try
                files = length(readdir(dir))
                println("   📄 Files: $files")
            catch
                println("   📄 Files: Unable to count")
            end
        end
    end
    
    # Check for trained model
    model_path = joinpath(CONFIG[:models_dir], CONFIG[:model_file])
    model_status = isfile(model_path) ? "✅ Available" : "❌ Not trained"
    println("🤖 Trained model: $model_status")
    
    # Check training data
    data_ok, msg = check_training_data()
    println("💾 Training data: $(data_ok ? "✅ Ready" : "⚠️ Not ready") - $msg")
    
    # Show available people
    people = list_people()
    if !isempty(people)
        println("👥 People in system: $(join(people, ", "))")
    end
    
    # Check visualization capabilities
    viz_status = check_visualization_system()
    println("🎨 Visualization system: $viz_status")
    
    println()
end

function check_visualization_system()
    try
        # Test if plotting capabilities are available
        test_fig = plot([1, 2, 3], [1, 4, 9], title="Test")
        return "✅ Ready (Plots.jl available)"
    catch
        return "⚠️ Limited (Plots.jl issues - visualizations may not work)"
    end
end

function show_main_menu()
    println("🎯 MAIN MENU:")
    println("-" ^ 15)
    println("1 - 📸 Capture photos from webcam")
    println("2 - 🧠 Train face recognition model")
    println("3 - 🔍 Test/predict with trained model")
    println("4 - 📊 System information")
    println("5 - 🗂️ Manage data (list/clean)")
    println("6 - 🎨 Visualization management")
    println("7 - ⚙️ Advanced options")
    println("8 - 🚪 Exit")
    println()
end

function show_advanced_menu()
    println("⚙️ ADVANCED OPTIONS:")
    println("-" ^ 20)
    println("1 - 🔬 Model architecture analysis")
    println("2 - 📈 Training history analysis") 
    println("3 - 🧹 Clean system (remove old files)")
    println("4 - 💾 Backup system data")
    println("5 - 🔄 Reset system (careful!)")
    println("6 - 🎛️ Configuration management")
    println("7 - ↩️ Back to main menu")
    println()
end

function show_visualization_menu()
    println("🎨 VISUALIZATION MANAGEMENT:")
    println("-" ^ 30)
    println("1 - 👁️ View existing visualizations")
    println("2 - 🖼️ Create visualizations for existing photos")
    println("3 - 📊 Generate training analysis plots")
    println("4 - 🗑️ Clean old visualizations")
    println("5 - 📤 Export visualization gallery")
    println("6 - ↩️ Back to main menu")
    println()
end

function capture_workflow()
    println("\n📸 PHOTO CAPTURE SYSTEM")
    println("=" ^ 25)
    
    success = capture_photos()
    
    if success
        println("\n✅ Photo capture completed!")
        
        # Check if we have enough data to train
        data_ok, msg = check_training_data()
        if data_ok
            print("🤖 Do you want to train the model now? (y/n): ")
            response = strip(lowercase(readline()))
            if response in ["y", "yes", "s", "sim"]
                return training_workflow()
            end
        else
            println("⚠️ Need more photos for training: $msg")
        end
    else
        println("❌ Photo capture failed!")
    end
    
    return success
end

function training_workflow()
    println("\n🧠 MODEL TRAINING SYSTEM")
    println("=" ^ 25)
    
    # Check prerequisites
    data_ok, msg = check_training_data()
    if !data_ok
        println("❌ Cannot train: $msg")
        println("Please capture photos first.")
        return false
    end
    
    println("Training prerequisites: $msg")
    print("Continue with training? (y/n): ")
    response = strip(lowercase(readline()))
    
    if response in ["y", "yes", "s", "sim"]
        success = train_model()
        
        if success
            println("\nTraining completed successfully!")
            println("You can now use option 3 to test the model")
            println("Layer visualizations have been created for all people")
        else
            println("Training failed. Please check the data and try again.")
        end
        
        return success
    else
        println("Training cancelled.")
        return false
    end
end

function test_model_workflow()
    println("\n🔍 MODEL TESTING SYSTEM")
    println("=" ^ 20)
    
    model_path = joinpath(CONFIG[:models_dir], CONFIG[:model_file])
    
    if !isfile(model_path)
        println("No trained model found!")
        println("Please train a model first (option 2)")
        return false
    end
    
    # Use the enhanced testing system from prediction.jl
    return test_model_system()
end

function show_system_info()
    println("\nSYSTEM INFORMATION")
    println("=" ^ 20)
    
    println("Version: $SYSTEM_VERSION")
    println("Configuration:")
    println("  Image size: $(CONFIG[:img_size])")
    println("  Batch size: $(CONFIG[:batch_size])")
    println("  Epochs: $(CONFIG[:epochs])")
    println("  Learning rate: $(CONFIG[:learning_rate])")
    
    println("\nDirectories:")
    println("  Data: $(CONFIG[:data_dir])")
    println("  Photos: $(CONFIG[:photos_dir])")
    println("  Models: $(CONFIG[:models_dir])")
    println("  Visualizations: $(CONFIG[:visualizations_dir])")
    
    println("\nFiles:")
    files_info = [
        (:model_file, "Model"),
        (:config_file, "Config"),
        (:weights_file, "Weights")
    ]
    
    for (key, filename) in files_info
        filepath = joinpath(CONFIG[:models_dir], CONFIG[key])
        status = isfile(filepath) ? "EXISTS" : "MISSING"
        size_info = ""
        if isfile(filepath)
            try
                size_kb = round(filesize(filepath) / 1024, digits=1)
                size_info = " ($(size_kb) KB)"
            catch
            end
        end
        println("  $filename: $status$size_info")
    end
    
    # Show model info if available
    config_path = joinpath(CONFIG[:models_dir], CONFIG[:config_file])
    if isfile(config_path)
        config = load_system_config()
        if config !== nothing
            println("\nTrained Model Info:")
            if haskey(config, "training")
                training = config["training"]
                println("  Best accuracy: $(round(get(training, "best_accuracy", 0.0)*100, digits=2))%")
                println("  Epochs trained: $(get(training, "epochs_trained", "unknown"))")
                println("  Duration: $(round(get(training, "duration_minutes", 0.0), digits=1)) minutes")
            end
            if haskey(config, "people")
                people = config["people"]
                println("  People count: $(get(people, "count", 0))")
                if haskey(people, "names")
                    println("  People: $(join(people["names"], ", "))")
                end
            end
        end
    end
    
    # Show visualization statistics
    show_visualization_statistics()
end

function show_visualization_statistics()
    try
        if isdir(CONFIG[:visualizations_dir])
            viz_dirs = readdir(CONFIG[:visualizations_dir])
            person_viz_count = count(d -> isdir(joinpath(CONFIG[:visualizations_dir], d)), viz_dirs)
            
            println("\nVisualization Statistics:")
            println("  Visualization directories: $person_viz_count")
            
            total_viz_files = 0
            for dir_name in viz_dirs
                dir_path = joinpath(CONFIG[:visualizations_dir], dir_name)
                if isdir(dir_path)
                    files = readdir(dir_path)
                    viz_files = count(f -> occursin(r"\.(png|jpg|jpeg)$", f), files)
                    total_viz_files += viz_files
                    if viz_files > 0
                        println("    $dir_name: $viz_files images")
                    end
                end
            end
            println("  Total visualization images: $total_viz_files")
        end
    catch e
        println("Error reading visualization statistics: $e")
    end
end

function manage_data()
    println("\nDATA MANAGEMENT")
    println("=" ^ 16)
    
    while true
        println("1 - List all photos")
        println("2 - List people and photo counts")
        println("3 - Delete photos for specific person")
        println("4 - Clean invalid photos")
        println("5 - Show detailed statistics")
        println("6 - Back to main menu")
        
        print("Choose option: ")
        option = strip(readline())
        
        if option == "1"
            list_all_photos()
        elseif option == "2"
            list_people()
        elseif option == "3"
            delete_person_photos()
        elseif option == "4"
            clean_invalid_photos()
        elseif option == "5"
            show_detailed_data_statistics()
        elseif option == "6"
            break
        else
            println("Invalid option")
        end
        
        println("\nPress Enter to continue...")
        readline()
    end
end

function visualization_management()
    println("\nVISUALIZATION MANAGEMENT")
    println("=" ^ 25)
    
    while true
        show_visualization_menu()
        print("Choose option: ")
        option = strip(readline())
        
        if option == "1"
            view_existing_visualizations()
        elseif option == "2"
            create_visualizations_for_existing_photos()
        elseif option == "3"
            generate_training_analysis_plots()
        elseif option == "4"
            clean_old_visualizations()
        elseif option == "5"
            export_visualization_gallery()
        elseif option == "6"
            break
        else
            println("Invalid option")
        end
        
        println("\nPress Enter to continue...")
        readline()
    end
end

function advanced_options()
    println("\nADVANCED OPTIONS")
    println("=" ^ 17)
    
    while true
        show_advanced_menu()
        print("Choose option: ")
        option = strip(readline())
        
        if option == "1"
            model_architecture_analysis()
        elseif option == "2"
            training_history_analysis()
        elseif option == "3"
            clean_system()
        elseif option == "4"
            backup_system_data()
        elseif option == "5"
            reset_system()
        elseif option == "6"
            configuration_management()
        elseif option == "7"
            break
        else
            println("Invalid option")
        end
        
        println("\nPress Enter to continue...")
        readline()
    end
end

# Advanced option implementations

function view_existing_visualizations()
    println("\nEXISTING VISUALIZATIONS")
    println("=" ^ 24)
    
    if !isdir(CONFIG[:visualizations_dir])
        println("No visualizations directory found")
        return
    end
    
    viz_dirs = readdir(CONFIG[:visualizations_dir])
    person_dirs = filter(d -> isdir(joinpath(CONFIG[:visualizations_dir], d)), viz_dirs)
    
    if isempty(person_dirs)
        println("No visualizations found")
        return
    end
    
    println("Available visualization directories:")
    for (i, dir_name) in enumerate(person_dirs)
        dir_path = joinpath(CONFIG[:visualizations_dir], dir_name)
        files = readdir(dir_path)
        image_count = count(f -> occursin(r"\.(png|jpg|jpeg)$", f), files)
        println("  $i. $dir_name ($image_count images)")
    end
    
    print("\nEnter number to view details (or press Enter to skip): ")
    choice = strip(readline())
    
    if !isempty(choice)
        try
            idx = parse(Int, choice)
            if 1 <= idx <= length(person_dirs)
                show_visualization_details(person_dirs[idx])
            end
        catch
            println("Invalid number")
        end
    end
end

function show_visualization_details(dir_name::String)
    dir_path = joinpath(CONFIG[:visualizations_dir], dir_name)
    
    println("\nVisualization details for: $dir_name")
    println("Directory: $dir_path")
    
    files = readdir(dir_path)
    image_files = filter(f -> occursin(r"\.(png|jpg|jpeg)$", f), files)
    text_files = filter(f -> occursin(r"\.txt$", f), files)
    
    println("Image files ($(length(image_files))):")
    for file in image_files
        println("  - $file")
    end
    
    if !isempty(text_files)
        println("Text files ($(length(text_files))):")
        for file in text_files
            println("  - $file")
        end
    end
end

function create_visualizations_for_existing_photos()
    println("\nCREATE VISUALIZATIONS FOR EXISTING PHOTOS")
    println("=" ^ 42)
    
    model_path = joinpath(CONFIG[:models_dir], CONFIG[:model_file])
    if !isfile(model_path)
        println("No trained model found. Please train a model first.")
        return
    end
    
    people = list_people()
    if isempty(people)
        println("No people found in photos directory")
        return
    end
    
    println("Available people: $(join(people, ", "))")
    print("Enter person name (or 'all' for everyone): ")
    target_person = strip(readline())
    
    try
        # Load model
        model_data = JLD2.load(model_path)
        model = model_data["model"]
        person_names = model_data["person_names"]
        
        if lowercase(target_person) == "all"
            create_all_people_visualizations(model, person_names)
        elseif target_person in people
            create_person_visualizations(model, person_names, target_person)
        else
            println("Person not found: $target_person")
        end
        
    catch e
        println("Error creating visualizations: $e")
    end
end

function create_all_people_visualizations(model, person_names)
    people = list_people()
    
    for person in people
        println("Creating visualizations for: $person")
        create_person_visualizations(model, person_names, person)
    end
    
    println("Visualizations created for all people!")
end

function create_person_visualizations(model, person_names, target_person::String)
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    person_photos = []
    
    # Find all photos for this person
    for filename in readdir(CONFIG[:photos_dir])
        ext = lowercase(splitext(filename)[2])
        if ext in image_extensions
            if extract_person_name(filename) == target_person
                push!(person_photos, joinpath(CONFIG[:photos_dir], filename))
            end
        end
    end
    
    if isempty(person_photos)
        println("No photos found for: $target_person")
        return
    end
    
    println("Found $(length(person_photos)) photos for $target_person")
    print("Create visualizations for all photos? (y/n): ")
    
    if lowercase(strip(readline())) in ["y", "yes"]
        create_batch_visualizations(person_photos, target_person)
    end
end

function generate_training_analysis_plots()
    println("\nGENERATE TRAINING ANALYSIS PLOTS")
    println("=" ^ 33)
    
    # Check if we have training data
    config_path = joinpath(CONFIG[:models_dir], CONFIG[:config_file])
    if !isfile(config_path)
        println("No training configuration found")
        return
    end
    
    config = load_system_config()
    if config === nothing
        println("Could not load training configuration")
        return
    end
    
    # Create comprehensive training analysis
    create_comprehensive_training_analysis(config)
    
    println("Training analysis plots generated!")
end

function create_comprehensive_training_analysis(config)
    try
        analysis_dir = joinpath(CONFIG[:visualizations_dir], "comprehensive_analysis")
        !isdir(analysis_dir) && mkpath(analysis_dir)
        
        training_info = config["training"]
        system_info = config["system"]
        
        # Create training overview plot
        fig = plot(layout=(2, 2), size=(1000, 800))
        
        # Plot 1: Training metrics (if available)
        if haskey(training_info, "training_losses") && haskey(training_info, "validation_accuracies")
            losses = training_info["training_losses"]
            accuracies = training_info["validation_accuracies"]
            
            plot!(fig[1], 1:length(losses), losses,
                  title="Training Loss", xlabel="Epoch", ylabel="Loss",
                  color=:red, linewidth=2)
            
            plot!(fig[2], 1:length(accuracies), accuracies .* 100,
                  title="Validation Accuracy", xlabel="Epoch", ylabel="Accuracy (%)",
                  color=:blue, linewidth=2)
        end
        
        # Plot 3: System information
        people_info = config["people"]
        bar!(fig[3], ["Classes", "Images", "Best Epoch"], 
             [people_info["count"], 
              get(training_info, "total_samples", 0),
              training_info["best_epoch"]],
             title="System Stats", color=:green, alpha=0.7)
        
        # Plot 4: Performance summary
        pie!(fig[4], [training_info["best_accuracy"], 1 - training_info["best_accuracy"]],
             labels=["Accuracy", "Error"], title="Model Performance")
        
        savefig(fig, joinpath(analysis_dir, "comprehensive_training_analysis.png"))
        
        # Save detailed text report
        create_detailed_text_report(config, analysis_dir)
        
    catch e
        println("Error creating comprehensive analysis: $e")
    end
end

function create_detailed_text_report(config, analysis_dir)
    report_path = joinpath(analysis_dir, "detailed_training_report.txt")
    
    open(report_path, "w") do io
        println(io, "DETAILED TRAINING REPORT")
        println(io, "=" ^ 25)
        println(io, "Generated: $(Dates.now())")
        println(io, "")
        
        # System information
        system_info = config["system"]
        println(io, "SYSTEM CONFIGURATION:")
        println(io, "  Version: $(system_info["version"])")
        println(io, "  Image Size: $(system_info["img_size"])")
        println(io, "  Number of Classes: $(system_info["num_classes"])")
        println(io, "")
        
        # Training results
        training_info = config["training"]
        println(io, "TRAINING RESULTS:")
        println(io, "  Best Accuracy: $(round(training_info["best_accuracy"]*100, digits=2))%")
        println(io, "  Best Epoch: $(training_info["best_epoch"])")
        println(io, "  Epochs Trained: $(training_info["epochs_trained"])")
        println(io, "  Duration: $(round(training_info["duration_minutes"], digits=2)) minutes")
        println(io, "")
        
        # People information
        people_info = config["people"]
        println(io, "PEOPLE IN SYSTEM:")
        for name in people_info["names"]
            println(io, "  - $name")
        end
        
        println(io, "\nTotal People: $(people_info["count"])")
    end
end

function clean_old_visualizations()
    println("\nCLEAN OLD VISUALIZATIONS")
    println("=" ^ 25)
    
    if !isdir(CONFIG[:visualizations_dir])
        println("No visualizations directory found")
        return
    end
    
    print("This will remove all visualization files. Continue? (yes/no): ")
    confirm = strip(lowercase(readline()))
    
    if confirm == "yes"
        try
            # Get total file count first
            total_files = count_visualization_files()
            
            # Remove all files
            rm(CONFIG[:visualizations_dir], recursive=true, force=true)
            mkpath(CONFIG[:visualizations_dir])
            
            println("Cleaned $total_files visualization files")
        catch e
            println("Error cleaning visualizations: $e")
        end
    else
        println("Cleaning cancelled")
    end
end

function count_visualization_files()
    total = 0
    try
        for (root, dirs, files) in walkdir(CONFIG[:visualizations_dir])
            total += length(files)
        end
    catch
    end
    return total
end

function export_visualization_gallery()
    println("\nEXPORT VISUALIZATION GALLERY")
    println("=" ^ 29)
    
    print("Enter export directory path: ")
    export_dir = strip(readline())
    
    if isempty(export_dir)
        export_dir = joinpath(CONFIG[:data_dir], "visualization_export")
    end
    
    try
        !isdir(export_dir) && mkpath(export_dir)
        
        # Copy all visualization files
        cp(CONFIG[:visualizations_dir], joinpath(export_dir, "visualizations"), force=true)
        
        # Create index HTML file
        create_gallery_html(export_dir)
        
        println("Gallery exported to: $export_dir")
        println("Open 'index.html' to view the gallery")
        
    catch e
        println("Error exporting gallery: $e")
    end
end

function create_gallery_html(export_dir)
    html_path = joinpath(export_dir, "index.html")
    
    open(html_path, "w") do io
        println(io, """
        <!DOCTYPE html>
        <html>
        <head>
            <title>CNN Face Recognition - Visualization Gallery</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .person-section { margin: 30px 0; border: 1px solid #ddd; padding: 20px; }
                .image-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }
                .image-item { text-align: center; }
                .image-item img { max-width: 100%; height: auto; border: 1px solid #ccc; }
            </style>
        </head>
        <body>
            <h1>CNN Face Recognition - Visualization Gallery</h1>
            <p>Generated: $(Dates.now())</p>
        """)
        
        # Add visualization sections
        viz_dir = joinpath(export_dir, "visualizations")
        if isdir(viz_dir)
            for person_dir in readdir(viz_dir)
                person_path = joinpath(viz_dir, person_dir)
                if isdir(person_path)
                    println(io, "<div class='person-section'>")
                    println(io, "<h2>$person_dir</h2>")
                    println(io, "<div class='image-grid'>")
                    
                    for file in readdir(person_path)
                        if occursin(r"\.(png|jpg|jpeg)$", file)
                            img_path = "visualizations/$person_dir/$file"
                            println(io, "<div class='image-item'>")
                            println(io, "<img src='$img_path' alt='$file'>")
                            println(io, "<p>$file</p>")
                            println(io, "</div>")
                        end
                    end
                    
                    println(io, "</div>")
                    println(io, "</div>")
                end
            end
        end
        
        println(io, """
        </body>
        </html>
        """)
    end
end

# Additional utility functions

function show_detailed_data_statistics()
    println("\nDETAILED DATA STATISTICS")
    println("=" ^ 25)
    
    if !isdir(CONFIG[:photos_dir])
        println("No photos directory found")
        return
    end
    
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    stats = Dict{String, Any}()
    
    # Collect statistics
    people_stats = Dict{String, Dict{String, Any}}()
    total_files = 0
    valid_files = 0
    invalid_files = 0
    
    for filename in readdir(CONFIG[:photos_dir])
        ext = lowercase(splitext(filename)[2])
        if ext in image_extensions
            filepath = joinpath(CONFIG[:photos_dir], filename)
            total_files += 1
            
            if validate_image_file(filepath)
                valid_files += 1
                person_name = extract_person_name(filename)
                
                if !haskey(people_stats, person_name)
                    people_stats[person_name] = Dict(
                        "count" => 0,
                        "files" => String[],
                        "total_size" => 0,
                        "extensions" => Set{String}()
                    )
                end
                
                people_stats[person_name]["count"] += 1
                push!(people_stats[person_name]["files"], filename)
                people_stats[person_name]["total_size"] += filesize(filepath)
                push!(people_stats[person_name]["extensions"], ext)
            else
                invalid_files += 1
            end
        end
    end
    
    # Display statistics
    println("OVERALL STATISTICS:")
    println("  Total image files: $total_files")
    println("  Valid images: $valid_files")
    println("  Invalid images: $invalid_files")
    println("  People count: $(length(people_stats))")
    
    println("\nPER-PERSON STATISTICS:")
    for (person, stats) in sort(collect(people_stats))
        size_mb = round(stats["total_size"] / (1024*1024), digits=2)
        extensions = join(collect(stats["extensions"]), ", ")
        
        println("  $person:")
        println("    Images: $(stats["count"])")
        println("    Total size: $(size_mb) MB")
        println("    File types: $extensions")
    end
end

function model_architecture_analysis()
    println("\nMODEL ARCHITECTURE ANALYSIS")
    println("=" ^ 28)
    
    model_path = joinpath(CONFIG[:models_dir], CONFIG[:model_file])
    if !isfile(model_path)
        println("No trained model found")
        return
    end
    
    try
        model_data = JLD2.load(model_path)
        model = model_data["model"]
        person_names = model_data["person_names"]
        
        println("Model loaded successfully")
        println("Classes: $(join(person_names, ", "))")
        
        # Analyze architecture
        analyze_model_architecture(model, person_names)
        
    catch e
        println("Error analyzing model: $e")
    end
end

function training_history_analysis()
    println("\nTRAINING HISTORY ANALYSIS")
    println("=" ^ 26)
    
    # This would analyze multiple training sessions if we had them stored
    println("Feature not yet implemented")
    println("This would show training history across multiple sessions")
end

function clean_system()
    println("\nSYSTEM CLEANUP")
    println("=" ^ 15)
    
    println("This will clean:")
    println("  - Invalid image files")
    println("  - Old temporary files")
    println("  - Empty directories")
    
    print("Continue? (y/n): ")
    if lowercase(strip(readline())) == "y"
        cleaned_count = 0
        
        # Clean invalid images
        cleaned_count += clean_invalid_photos_silent()
        
        # Clean temp files
        try
            temp_files = filter(f -> occursin("temp", f), readdir(tempdir()))
            for file in temp_files
                try
                    rm(joinpath(tempdir(), file))
                    cleaned_count += 1
                catch
                end
            end
        catch
        end
        
        println("Cleanup completed. Removed $cleaned_count items.")
    end
end

function clean_invalid_photos_silent()
    # Silent version of clean_invalid_photos for system cleanup
    if !isdir(CONFIG[:photos_dir])
        return 0
    end
    
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    cleaned = 0
    
    for filename in readdir(CONFIG[:photos_dir])
        ext = lowercase(splitext(filename)[2])
        if ext in image_extensions
            filepath = joinpath(CONFIG[:photos_dir], filename)
            if !validate_image_file(filepath)
                try
                    rm(filepath)
                    cleaned += 1
                catch
                end
            end
        end
    end
    
    return cleaned
end

function backup_system_data()
    println("\nSYSTEM BACKUP")
    println("=" ^ 14)
    
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    backup_dir = joinpath(dirname(CONFIG[:data_dir]), "webcamcnn_backup_$timestamp")
    
    try
        println("Creating backup at: $backup_dir")
        cp(CONFIG[:data_dir], backup_dir)
        println("Backup completed successfully!")
        
        # Show backup size
        backup_size = 0
        for (root, dirs, files) in walkdir(backup_dir)
            for file in files
                backup_size += filesize(joinpath(root, file))
            end
        end
        
        size_mb = round(backup_size / (1024*1024), digits=2)
        println("Backup size: $(size_mb) MB")
        
    catch e
        println("Backup failed: $e")
    end
end

function reset_system()
    println("\nSYSTEM RESET")
    println("=" ^ 13)
    
    println("WARNING: This will delete ALL data including:")
    println("  - All photos")
    println("  - Trained models")
    println("  - Visualizations") 
    println("  - Configuration files")
    
    print("Type 'RESET' to confirm: ")
    confirm = strip(readline())
    
    if confirm == "RESET"
        try
            rm(CONFIG[:data_dir], recursive=true, force=true)
            init_directories()
            println("System reset completed!")
        catch e
            println("Reset failed: $e")
        end
    else
        println("Reset cancelled")
    end
end

function configuration_management()
    println("\nCONFIGURATION MANAGEMENT")
    println("=" ^ 25)
    
    println("Current configuration:")
    for (key, value) in CONFIG
        if key != :data_dir  # Skip paths for cleaner display
            println("  $key: $value")
        end
    end
    
    println("\nConfiguration management features not yet implemented")
    println("Future versions will allow modifying training parameters")
end

# Standard utility functions

function list_all_photos()
    if !isdir(CONFIG[:photos_dir])
        println("No photos directory found")
        return
    end
    
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    photos = []
    
    for filename in readdir(CONFIG[:photos_dir])
        ext = lowercase(splitext(filename)[2])
        if ext in image_extensions
            filepath = joinpath(CONFIG[:photos_dir], filename)
            person = extract_person_name(filename)
            valid = validate_image_file(filepath)
            push!(photos, (filename, person, valid))
        end
    end
    
    if isempty(photos)
        println("No photos found")
        return
    end
    
    println("All photos ($(length(photos)) total):")
    for (filename, person, valid) in photos
        status = valid ? "OK" : "INVALID"
        println("  $filename -> $person [$status]")
    end
end

function delete_person_photos()
    people = list_people()
    if isempty(people)
        println("No people found")
        return
    end
    
    print("Enter person name to delete photos: ")
    target_person = strip(readline())
    
    if target_person ∉ people
        println("Person '$target_person' not found")
        return
    end
    
    print("Are you sure you want to delete ALL photos for '$target_person'? (yes/no): ")
    confirm = strip(lowercase(readline()))
    
    if confirm != "yes"
        println("Deletion cancelled")
        return
    end
    
    deleted_count = 0
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    
    for filename in readdir(CONFIG[:photos_dir])
        ext = lowercase(splitext(filename)[2])
        if ext in image_extensions
            if extract_person_name(filename) == target_person
                filepath = joinpath(CONFIG[:photos_dir], filename)
                try
                    rm(filepath)
                    deleted_count += 1
                    println("Deleted: $filename")
                catch e
                    println("Error deleting $filename: $e")
                end
            end
        end
    end
    
    println("Deleted $deleted_count photos for '$target_person'")
    
    # Also clean up visualizations for this person
    person_viz_dir = joinpath(CONFIG[:visualizations_dir], target_person)
    if isdir(person_viz_dir)
        try
            rm(person_viz_dir, recursive=true)
            println("Also removed visualizations for '$target_person'")
        catch e
            println("Could not remove visualizations: $e")
        end
    end
end

function clean_invalid_photos()
    if !isdir(CONFIG[:photos_dir])
        println("No photos directory found")
        return
    end
    
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    invalid_photos = []
    
    for filename in readdir(CONFIG[:photos_dir])
        ext = lowercase(splitext(filename)[2])
        if ext in image_extensions
            filepath = joinpath(CONFIG[:photos_dir], filename)
            if !validate_image_file(filepath)
                push!(invalid_photos, filepath)
            end
        end
    end
    
    if isempty(invalid_photos)
        println("No invalid photos found")
        return
    end
    
    println("Found $(length(invalid_photos)) invalid photos:")
    for photo in invalid_photos
        println("  - $(basename(photo))")
    end
    
    print("Delete all invalid photos? (y/n): ")
    if lowercase(strip(readline())) == "y"
        for photo in invalid_photos
            try
                rm(photo)
                println("Deleted: $(basename(photo))")
            catch e
                println("Error deleting $(basename(photo)): $e")
            end
        end
        println("Cleanup completed")
    else
        println("Cleanup cancelled")
    end
end

# Main execution function
function main()
    # Initialize system
    try
        init_directories()
    catch e
        println("Error initializing directories: $e")
        return 1
    end
    
    print_header()
    print_system_status()
    
    while true
        show_main_menu()
        print("Choose option (1-8): ")
        option = strip(readline())
        
        if option == "1"
            capture_workflow()
        elseif option == "2"
            training_workflow()
        elseif option == "3"
            test_model_workflow()
        elseif option == "4"
            show_system_info()
        elseif option == "5"
            manage_data()
        elseif option == "6"
            visualization_management()
        elseif option == "7"
            advanced_options()
        elseif option == "8"
            println("\nExiting system...")
            println("Thank you for using Enhanced CNN Face Recognition System!")
            break
        else
            println("Invalid option. Please choose 1-8.")
        end
        
        println("\nPress Enter to return to main menu...")
        readline()
        println("\n" * "=" ^ 65)
    end
    
    return 0
end

# Execute main function if script is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end