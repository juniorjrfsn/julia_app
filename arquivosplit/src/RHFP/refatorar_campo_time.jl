# Projeto : arquivosplit
# Arquivo : src/refatorar_campo_time.jl
#
# Objetivo: Refatorar o campo de tempo (HORA) de arquivos de dados separados por "|",
#           convertendo o formato HH.MM.SS para HH:MM:SS.
# ─────────────────────────────────────────────────────────────────────────────

function convert_file(path::String)::Bool
    if !isfile(path)
        @warn "Arquivo não encontrado: $path"
        return false
    end

    println("Lendo arquivo $path...")
    content = read(path, String)
    # Detecta quebra de linha legítima (CRLF vs LF)
    newline = occursin("\r\n", content) ? "\r\n" : "\n"
    lines = split(content, newline)

    modified = false
    new_lines = String[]
    sizehint!(new_lines, length(lines))
    
    # Índice da coluna de tempo (1-based), detectado dinamicamente no cabeçalho
    time_col_idx = 0

    for (idx, line) in enumerate(lines)
        # Se for a última linha e estiver vazia, preserva-a
        if idx == length(lines) && isempty(line)
            push!(new_lines, "")
            continue
        end

        # Processa o cabeçalho
        if idx == 1
            parts = split(line, '|')
            for (col_idx, name) in enumerate(parts)
                if occursin(r"HORA"i, name)
                    time_col_idx = col_idx
                    break
                end
            end
            push!(new_lines, String(line))
            continue
        end

        parts = split(line, '|')
        if time_col_idx > 0 && length(parts) >= time_col_idx
            field = parts[time_col_idx]
            # Verifica se o campo de tempo está no formato HH.MM.SS (por exemplo, 15.33.57 ou 9.51.56)
            if occursin(r"^\d{1,2}\.\d{2}\.\d{2}$", field)
                new_field = replace(field, '.' => ':')
                if new_field != field
                    parts[time_col_idx] = new_field
                    modified = true
                end
            end
        end
        push!(new_lines, join(parts, '|'))
    end

    if !modified
        println("  [ok]        $(basename(path)) (sem alterações)")
        return false
    end

    # Cria backup antes de sobrescrever
    backup = path * ".bak"
    println("Criando backup em $backup...")
    cp(path, backup; force=true)

    println("Gravando alterações...")
    open(path, "w") do io
        print(io, join(new_lines, newline))
    end
    println("  [refatorado] $(basename(path)) — backup: $backup")
    return true
end

# ── Processamento de pasta ────────────────────────────────────────────────────

function process_folder(dir::String)
    if !isdir(dir)
        @error "Diretório não encontrado: $dir"
        exit(1)
    end

    println("Diretório: $dir\n")
    files = filter(isfile, readdir(dir, join=true))

    if isempty(files)
        println("Nenhum arquivo encontrado.")
        return
    end

    converted = 0
    for file in files
        # Apenas processa arquivos .txt
        if endswith(lowercase(file), ".txt")
            converted += convert_file(file) ? 1 : 0
        end
    end

    println("\nConcluído: $converted arquivo(s) refatorado(s).")
end

# ── Ajuda ─────────────────────────────────────────────────────────────────────

function print_help()
    println("""
    Uso:
      julia refatorar_campo_time.jl
            Processa os arquivos padrões (TRAMITE_PROCESSO.txt e REVALIDACAO_CIS.txt)

      julia refatorar_campo_time.jl <arquivo>
            Refatora o formato de hora em um arquivo específico

      julia refatorar_campo_time.jl <diretório>
            Refatora o formato de hora em todos os arquivos .txt na pasta

      julia refatorar_campo_time.jl -h | --help
            Exibe esta ajuda

    Comportamento:
      • Identifica dinamicamente a coluna de hora (que contenha "HORA" no cabeçalho)
      • Converte o formato HH.MM.SS para HH:MM:SS
      • Cria backup <arquivo>.bak antes de modificar
    """)
end

# ── Ponto de entrada ──────────────────────────────────────────────────────────

args = ARGS

if length(args) >= 1 && args[1] in ("-h", "--help")
    print_help()
    exit(0)
end

# Tabelas e arquivos padrões de destino
const DEFAULT_TARGETS = [
    raw"C:\Users\njunior\Documents\RHFP\TABELA\TRAMITE_PROCESSO.txt",
    raw"C:\Users\njunior\Documents\RHFP\TABELA\REVALIDACAO_CIS.txt",
    raw"C:\Users\njunior\Documents\RHFP\TABELA\TROCA_MATRICULA.txt"
]

if length(args) >= 1
    target = args[1]
    if isfile(target)
        println("Arquivo: $target\n")
        convert_file(target)
        println("\nConcluído.")
    elseif isdir(target)
        process_folder(target)
    else
        @error "Caminho não encontrado: $target"
        println()
        print_help()
        exit(1)
    end
else
    println("Processando arquivos padrões...\n")
    for target in DEFAULT_TARGETS
        if isfile(target)
            convert_file(target)
            println()
        else
            @warn "Arquivo padrão não encontrado: $target"
            println()
        end
    end
    println("Concluído.")
end

# julia ./arquivosplit/src/refatorar_campo_time.jl
