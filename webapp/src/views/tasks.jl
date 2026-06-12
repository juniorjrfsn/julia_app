function escape_html(s::String)::String
    s = replace(s, "&"  => "&amp;")
    s = replace(s, "<"  => "&lt;")
    s = replace(s, ">"  => "&gt;")
    s = replace(s, '"' => "&quot;")
    s
end

function render_tasks_index(tasks::Vector{Task})::String
    total = length(tasks)
    ndone = count(t -> t.done, tasks)

    stats = """
    <div class="stats">
    <div>total <strong>$total</strong></div>
    <div>concluídas <strong>$ndone</strong></div>
    <div>pendentes <strong>$(total - ndone)</strong></div>
    </div>"""

    form = """
    <form class="form-row" action="/tasks" method="POST">
    <input type="text" name="title" placeholder="Nova tarefa..." required autofocus/>
    <button class="btn" type="submit">+ add</button>
    </form>"""

    items = if isempty(tasks)
        """<div class="empty"><span>📭</span>Nenhuma tarefa ainda.</div>"""
    else
        rows = join(["""
    <li class="task-item $(t.done ? "done" : "")">
        <input class="task-check" type="checkbox" data-id="$(t.id)"$(t.done ? " checked" : "")/>
        <span class="task-title">$(escape_html(t.title))</span>
        <span class="task-meta">$(Dates.format(t.created_at, "dd/mm HH:MM"))</span>
        <div class="actions">
        <form action="/tasks/$(t.id)/delete" method="POST" style="margin:0">
            <button class="btn btn-sm btn-danger" type="submit">✕</button>
        </form>
        </div>
    </li>""" for t in tasks])
        "<ul class=\"task-list\">" * rows * "</ul>"
    end

    page = render_template(
        read_template("tasks.html"),
        Dict("stats" => stats, "form" => form, "items" => items)
    )

    render_layout("tasks // julia mvc", page)
end
