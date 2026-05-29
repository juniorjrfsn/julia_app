module WebApp

using HTTP
using JSON3
using Dates

include("model.jl")
include("views/layout.jl")
include("views/tasks.jl")
include("controller.jl")
include("server.jl")

export start

end # module WebApp
