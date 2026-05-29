mutable struct Task
    id::Int
    title::String
    done::Bool
    created_at::DateTime
end

const DB      = Dict{Int, Task}()
const NEXT_ID = Ref(1)

function all_tasks()::Vector{Task}
    sort(collect(values(DB)), by = t -> t.id)
end

function find_task(id::Int)::Union{Task, Nothing}
    get(DB, id, nothing)
end

function create_task(title::String)::Task
    id = NEXT_ID[]
    t  = Task(id, title, false, now())
    DB[id] = t
    NEXT_ID[] += 1
    t
end

function update_task(id::Int;
                    title::Union{String, Nothing} = nothing,
                    done::Union{Bool, Nothing} = nothing)::Union{Task, Nothing}
    t = find_task(id)
    t === nothing && return nothing
    title !== nothing && (t.title = title)
    done  !== nothing && (t.done  = done)
    t
end

function delete_task(id::Int)::Bool
    haskey(DB, id) ? (delete!(DB, id); true) : false
end
