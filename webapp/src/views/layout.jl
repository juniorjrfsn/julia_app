function read_template(name::String)::String
    read(joinpath(@__DIR__, name), String)
end

function render_template(template::String, replacements::Dict{String, String})::String
    result = template
    for (key, value) in replacements
        result = replace(result, "{{" * key * "}}" => value)
    end
    result
end

function render_layout(title::String, body::String)::String
    template = read_template("layout.html")
    render_template(template, Dict("title" => title, "body" => body))
end
