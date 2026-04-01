open("debug_error.log", "w") do f
    try
        include("src/webcamcnnwindows.jl")
        Base.invokelatest(webcamcnnwindows.run_app)
        println(f, "Ran Successfully")
    catch e
        showerror(f, e, catch_backtrace())
    end
end
