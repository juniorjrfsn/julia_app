# src/gui.jl

using Gtk4, VideoIO, Images, Dates, Cairo

# Global state for GUI
mutable struct AppState
    camera_running::Bool
    camera::Union{Nothing, VideoIO.VideoReader}
    current_frame::Union{Nothing, Matrix{RGB{N0f8}}}
    person_name::String
    model::Any
    person_names::Vector{String}
    update_task::Union{Nothing, Task}
    
    AppState() = new(false, nothing, nothing, "", nothing, String[], nothing)
end

const state = AppState()

function start_gui()
    win = GtkWindow("CNN Face Recognition", 900, 700)
    
    # Notebook provides tabs
    nb = GtkNotebook()
    push!(win, nb)

    # Init UI components
    page_capture = build_capture_ui()
    page_train = build_train_ui()
    page_predict = build_predict_ui()
    
    append_page!(nb, page_capture, GtkLabel("📷 Capture"))
    append_page!(nb, page_train, GtkLabel("🧠 Train"))
    append_page!(nb, page_predict, GtkLabel("🔍 Predict"))

    # Load model if exists
    load_model_for_predict()

    # Window close event
    signal_connect(win, "close-request") do widget
        stop_camera()
        return false
    end

    show(win)
    
    if !isinteractive()
        cond = Condition()
        signal_connect(win, "close-request") do widget
            notify(cond)
            return false
        end
        wait(cond)
    end
end

function build_capture_ui()
    box = GtkBox(:v, 10)
    set_margin_start(box, 10); set_margin_end(box, 10)
    set_margin_top(box, 10); set_margin_bottom(box, 10)
    
    controls = GtkBox(:h, 10)
    push!(box, controls)
    
    btn_start_cam = GtkButton("Start Camera")
    btn_stop_cam = GtkButton("Stop Camera")
    push!(controls, btn_start_cam)
    push!(controls, btn_stop_cam)
    
    lbl_name = GtkLabel("Person Name: ")
    entry_name = GtkEntry()
    push!(controls, lbl_name)
    push!(controls, entry_name)
    
    btn_capture = GtkButton("📸 Take Photo")
    push!(controls, btn_capture)
    
    lbl_status = GtkLabel("Status: Ready")
    push!(box, lbl_status)
    
    # Canvas area
    canvas = GtkCanvas()
    canvas.hexpand = true
    canvas.vexpand = true
    push!(box, canvas)
    
    signal_connect(btn_start_cam, "clicked") do widget
        start_camera(canvas, lbl_status)
    end
    
    signal_connect(btn_stop_cam, "clicked") do widget
        stop_camera()
        lbl_status.label = "Status: Camera stopped"
    end
    
    signal_connect(btn_capture, "clicked") do widget
        name = entry_name.text
        if isempty(name)
            lbl_status.label = "Status: Please enter a name first."
            return
        end
        if state.current_frame !== nothing
            save_photo(name, state.current_frame, lbl_status)
        else
            lbl_status.label = "Status: No frame to capture."
        end
    end
    
    # Drawing function
    @guarded draw(canvas) do widget
        ctx = getgc(canvas)
        w = width(canvas)
        h = height(canvas)
        
        # Clear background
        rectangle(ctx, 0, 0, w, h)
        set_source_rgb(ctx, 0.1, 0.1, 0.1)
        fill(ctx)
        
        if state.current_frame !== nothing
            # Convert frame to Cairo image surface format
            img = state.current_frame
            img_h, img_w = size(img)
            
            # Simple scaling to fit canvas while preserving aspect ratio
            scale_factor = min(w / img_w, h / img_h)
            new_w = img_w * scale_factor
            new_h = img_h * scale_factor
            
            x_offset = (w - new_w) / 2
            y_offset = (h - new_h) / 2
            
            # Convert to ARGB32 that Cairo requires
            img_argb = collect(colorview(ARGB32, img))
            surf = Cairo.CairoImageSurface(img_argb)
            
            Cairo.save(ctx)
            Cairo.translate(ctx, x_offset, y_offset)
            Cairo.scale(ctx, scale_factor, scale_factor)
            Cairo.set_source_surface(ctx, surf, 0, 0)
            Cairo.paint(ctx)
            Cairo.restore(ctx)
        else
            # Draw empty state text
            move_to(ctx, w/2 - 40, h/2)
            set_source_rgb(ctx, 1.0, 1.0, 1.0)
            show_text(ctx, "Camera Off")
        end
    end
    
    return box
end

function build_train_ui()
    box = GtkBox(:v, 10)
    set_margin_start(box, 10); set_margin_end(box, 10)
    set_margin_top(box, 10); set_margin_bottom(box, 10)
    
    lbl_info = GtkLabel("Click 'Train Model' to start training on captured photos.")
    push!(box, lbl_info)
    
    btn_train = GtkButton("🧠 Train Model")
    push!(box, btn_train)
    
    lbl_status = GtkLabel("Status: Ready")
    push!(box, lbl_status)
    
    txt_log = GtkTextView()
    txt_log.editable = false
    txt_log.hexpand = true
    txt_log.vexpand = true
    scrolled = GtkScrolledWindow()
    push!(scrolled, txt_log)
    push!(box, scrolled)
    
    signal_connect(btn_train, "clicked") do widget
        # Execute training async to avoid freezing GUI
        lbl_status.label = "Status: Training (Processing...)"
        btn_train.sensitive = false
        
        @async begin
            try
                success = train_model_gui_wrapper(txt_log)
                if success
                    lbl_status.label = "Status: Training Complete!"
                    load_model_for_predict()
                else
                    lbl_status.label = "Status: Training Failed!"
                end
            catch e
                lbl_status.label = "Status: Error ($e)"
            finally
                btn_train.sensitive = true
            end
        end
    end
    
    return box
end

function build_predict_ui()
    box = GtkBox(:v, 10)
    set_margin_start(box, 10); set_margin_end(box, 10)
    set_margin_top(box, 10); set_margin_bottom(box, 10)
    
    controls = GtkBox(:h, 10)
    push!(box, controls)
    
    btn_start_cam = GtkButton("Start Camera")
    btn_stop_cam = GtkButton("Stop Camera")
    push!(controls, btn_start_cam)
    push!(controls, btn_stop_cam)
    
    lbl_result = GtkLabel("Prediction: None")
    # Increase font size for prediction
    # ...
    push!(box, lbl_result)
    
    # Canvas area
    canvas = GtkCanvas()
    canvas.hexpand = true
    canvas.vexpand = true
    push!(box, canvas)
    
    signal_connect(btn_start_cam, "clicked") do widget
        if state.model === nothing
            lbl_result.label = "Error: No trained model available."
            return
        end
        start_camera(canvas, lbl_result, predict=true)
    end
    
    signal_connect(btn_stop_cam, "clicked") do widget
        stop_camera()
        lbl_result.label = "Camera stopped"
    end
    
    @guarded draw(canvas) do widget
        ctx = getgc(canvas)
        w = width(canvas)
        h = height(canvas)
        
        rectangle(ctx, 0, 0, w, h)
        set_source_rgb(ctx, 0.1, 0.1, 0.1)
        fill(ctx)
        
        if state.current_frame !== nothing
            img = state.current_frame
            img_h, img_w = size(img)
            
            scale_factor = min(w / img_w, h / img_h)
            new_w = img_w * scale_factor
            new_h = img_h * scale_factor
            
            x_offset = (w - new_w) / 2
            y_offset = (h - new_h) / 2
            
            img_argb = collect(colorview(ARGB32, img))
            surf = Cairo.CairoImageSurface(img_argb)
            
            Cairo.save(ctx)
            Cairo.translate(ctx, x_offset, y_offset)
            Cairo.scale(ctx, scale_factor, scale_factor)
            Cairo.set_source_surface(ctx, surf, 0, 0)
            Cairo.paint(ctx)
            Cairo.restore(ctx)
        end
    end
    
    return box
end

# --- Camera controls ---
function start_camera(canvas, status_label; predict=false)
    if state.camera_running
        return
    end
    
    try
        state.camera = VideoIO.opencamera()
        state.camera_running = true
        status_label.label = "Status: Camera Active"
        
        # Start update task
        state.update_task = @async while state.camera_running
            frame = read(state.camera)
            if frame !== nothing
                state.current_frame = frame
                
                # Update canvas in GUI thread safely
                # Gtk4 canvas needs to be marked dirty
                Gtk4.GLib.g_idle_add() do
                    reveal(canvas)
                    Cint(0) # GL_SOURCE_REMOVE
                end
                
                if predict && state.model !== nothing
                    # Do prediction occasionally
                    if rand() < 0.2 # 5fps roughly to not freeze
                        pred, conf = predict_frame(frame)
                        Gtk4.GLib.g_idle_add() do
                            status_label.label = pred !== nothing ? "Prediction: $pred ($(round(conf*100, digits=1))%)" : "Prediction: None"
                            Cint(0)
                        end
                    end
                end
            end
            sleep(0.03) # ~30 FPS
        end
    catch e
        status_label.label = "Status: Error opening camera ($e)"
    end
end

function stop_camera()
    state.camera_running = false
    if state.camera !== nothing
        close(state.camera)
        state.camera = nothing
    end
    state.current_frame = nothing
end

function save_photo(name, frame, label)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    filename = "$(name)_gui_$timestamp.jpg"
    filepath = joinpath(CONFIG[:photos_dir], filename)
    save(filepath, frame)
    label.label = "Status: Photo saved -> $filename"
end

function load_model_for_predict()
    model_path = joinpath(CONFIG[:models_dir], CONFIG[:model_file])
    
    if isfile(model_path)
        try
            data = JLD2.load(model_path)
            state.model = data["model"]
            state.person_names = data["person_names"]
            println("Model loaded for prediction.")
        catch e
            println("Error loading model: $e")
        end
    end
end

function predict_frame(frame)
    # Save temp, then process
    temp_path = joinpath(tempdir(), "temp_pred.jpg")
    save(temp_path, frame)
    
    pred, conf = predict_person(state.model, state.person_names, temp_path)
    try; rm(temp_path); catch; end
    return pred, conf
end

function train_model_gui_wrapper(txt_log)
    buffer = Gtk4.buffer(txt_log)
    Gtk4.text(buffer, "Starting training...\n")
    
    data_ok, msg = check_training_data()
    if !data_ok
        Gtk4.GLib.g_idle_add() do
            Gtk4.text(buffer, Gtk4.text(buffer) * "Error: $msg\n")
            Cint(0)
        end
        return false
    end
    
    # Redirect stdout to capture training logs
    original_stdout = stdout
    rd, wr = redirect_stdout()
    
    # Task to read from the redirected stream and update GUI
    log_task = @async begin
        while isopen(rd) && !eof(rd)
            line = readline(rd)
            Gtk4.GLib.g_idle_add() do
                Gtk4.text(buffer, Gtk4.text(buffer) * line * "\n")
                
                # Scroll to bottom
                adj = Gtk4.vadjustment(txt_log)
                if adj !== nothing
                    Gtk4.value(adj, Gtk4.upper(adj) - Gtk4.page_size(adj))
                end
                
                Cint(0)
            end
        end
    end
    
    result = false
    try
        result = train_model()
    catch e
        println("Training exception: $e")
    finally
        redirect_stdout(original_stdout)
        close(wr)
        # Wait a small bit to allow reader to finish
        sleep(0.1)
    end
    
    return result
end
