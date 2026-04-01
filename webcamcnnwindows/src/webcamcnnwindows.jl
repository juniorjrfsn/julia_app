module webcamcnnwindows

# Internal logic
include("config.jl")
include("capture.jl")
include("training.jl")
include("prediction.jl")

# GUI logic
include("gui.jl")

function run_app()
    println("Initializing CNN Face Recognition GUI...")
    init_directories()
    start_gui()
end

# Se chamado diretamente como script
if abspath(PROGRAM_FILE) == @__FILE__
    run_app()
end

end # module webcamcnnwindows
