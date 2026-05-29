function html_response(body::String; status::Int = 200)::HTTP.Response
    HTTP.Response(status, ["Content-Type" => "text/html; charset=utf-8"], body)
end

function json_response(data; status::Int = 200)::HTTP.Response
    HTTP.Response(status, ["Content-Type" => "application/json"],
                String(JSON3.write(data)))
end

function redirect(location::String)::HTTP.Response
    HTTP.Response(302, ["Location" => location], "")
end

function parse_form(req::HTTP.Request)::Dict{String, String}
    body = String(req.body)
    result = Dict{String, String}()
    for part in split(body, "&")
        idx = findfirst('=', part)
        idx === nothing && continue
        k = String(part[1:idx-1])
        v = HTTP.URIs.unescapeuri(String(part[idx+1:end]))
        result[k] = replace(v, "+" => " ")
    end
    result
end

function route(req::HTTP.Request)::HTTP.Response
    method = req.method
    path   = HTTP.URI(req.target).path

    method == "GET" && path == "/" && return redirect("/tasks")

    method == "GET" && path == "/tasks" &&
        return html_response(render_tasks_index(all_tasks()))

    if method == "POST" && path == "/tasks"
        params = parse_form(req)
        title  = strip(get(params, "title", ""))
        isempty(title) && return redirect("/tasks")
        create_task(String(title))
        return redirect("/tasks")
    end

    m = match(r"^/tasks/(\d+)/toggle$", path)
    if m !== nothing && method == "POST"
        id = parse(Int, m.captures[1])
        t  = find_task(id)
        if t !== nothing
            update_task(id; done = !t.done)
        end
        return redirect("/tasks")
    end

    m = match(r"^/tasks/(\d+)/delete$", path)
    if m !== nothing && method == "POST"
        delete_task(parse(Int, m.captures[1]))
        return redirect("/tasks")
    end

    method == "GET" && path == "/api/tasks" &&
        return json_response(all_tasks())

    m = match(r"^/api/tasks/(\d+)$", path)
    if m !== nothing && method == "GET"
        t = find_task(parse(Int, m.captures[1]))
        t === nothing &&
            return json_response(Dict("error" => "not found"); status = 404)
        return json_response(t)
    end

    if method == "POST" && path == "/api/tasks"
        data  = JSON3.read(String(req.body))
        title = strip(get(data, :title, ""))
        isempty(title) &&
            return json_response(Dict("error" => "title required"); status = 400)
        return json_response(create_task(String(title)); status = 201)
    end

    m = match(r"^/api/tasks/(\d+)$", path)
    if m !== nothing && method == "DELETE"
        ok = delete_task(parse(Int, m.captures[1]))
        return ok ? json_response(Dict("ok" => true)) :
                    json_response(Dict("error" => "not found"); status = 404)
    end

    html_response("<h1>404 — não encontrado</h1>"; status = 404)
end
