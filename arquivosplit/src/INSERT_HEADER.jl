function load_header_mapping(headers_json::AbstractString)
    text = read(headers_json, String)
    mapping = Dict{String,String}()
    regex = r"\{\s*\"NOME_DA_TABELA\"\s*:\s*\"(?<name>[^\"]+)\"\s*,\s*\"HEADER\"\s*:\s*\"(?<header>[^\"]*)\"\s*\}"s

    for m in eachmatch(regex, text)
        name = m.captures[1]
        header = m.captures[2]
        mapping[name] = header
    end

    return mapping
end

function find_table_file(table_name::AbstractString; tabela_dir::AbstractString="TABELA")
    tabela_dir = normpath(tabela_dir)
    if !isdir(tabela_dir)
        error("Diretório de tabela não existe: $tabela_dir")
    end

    candidate = joinpath(tabela_dir, table_name)
    if isfile(candidate)
        return candidate
    end

    for entry in readdir(tabela_dir)
        path = joinpath(tabela_dir, entry)
        if isfile(path)
            base = splitext(entry)[1]
            if base == table_name
                return path
            end
        end
    end

    error("Arquivo de tabela não encontrado em '$tabela_dir' para NOME_DA_TABELA='$table_name'")
end

function insert_header_in_file!(file_path::AbstractString, header::AbstractString)
    content = read(file_path, String)
    lines = split(content, '\n', keepempty=true)
    if !isempty(lines) && chomp(lines[1]) == header
        println("Cabeçalho já presente em: $file_path")
        return false
    end

    new_content = header * "\n" * content
    write(file_path, new_content)
    println("Cabeçalho inserido com sucesso em: $file_path")
    return true
end

function insert_header!(table_name::AbstractString;
                        tabela_dir::AbstractString=raw"C:\Users\njunior\Documents\RHFP\TABELA",
                        headers_json::AbstractString=raw"C:\Users\njunior\Documents\RHFP\MSPREV_HEADERS.json")
    mapping = load_header_mapping(headers_json)
    header = get(mapping, table_name, nothing)
    if header === nothing
        error("Cabeçalho não encontrado em $headers_json para NOME_DA_TABELA='$table_name'")
    end

    file_path = find_table_file(table_name; tabela_dir=tabela_dir)
    return insert_header_in_file!(file_path, header)
end

function insert_headers_all!(;
                              tabela_dir::AbstractString=raw"C:\Users\njunior\Documents\RHFP\TABELA",
                              headers_json::AbstractString=raw"C:\Users\njunior\Documents\RHFP\MSPREV_HEADERS.json")
    mapping = load_header_mapping(headers_json)
    tabela_dir = normpath(tabela_dir)
    if !isdir(tabela_dir)
        error("Diretório de tabela não existe: $tabela_dir")
    end

    files = sort(readdir(tabela_dir))
    inserted = 0
    skipped = 0
    missing = 0

    for entry in files
        path = joinpath(tabela_dir, entry)
        if !isfile(path)
            continue
        end

        name = splitext(entry)[1]
        header = get(mapping, name, nothing)
        if header === nothing
            println("Sem cabeçalho definido para: $entry")
            missing += 1
            continue
        end

        if insert_header_in_file!(path, header)
            inserted += 1
        else
            skipped += 1
        end
    end

    println("\nResumo: inseridos=$inserted, pulados=$skipped, sem_mapeamento=$missing")
    return inserted
end

function main(args::Vector{String}=ARGS)
    tabela_dir = raw"C:\Users\njunior\Documents\RHFP\TABELA"
    headers_json = raw"C:\Users\njunior\Documents\RHFP\MSPREV_HEADERS.json"

    if length(args) == 0 || uppercase(args[1]) == "ALL"
        insert_headers_all!(tabela_dir=tabela_dir, headers_json=headers_json)
        return
    end

    table_name = args[1]
    if length(args) >= 2
        tabela_dir = args[2]
    end
    if length(args) >= 3
        headers_json = args[3]
    end

    insert_header!(table_name; tabela_dir=tabela_dir, headers_json=headers_json)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end


# julia arquivosplit/src/INSERT_HEADER.jl