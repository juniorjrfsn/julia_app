# Projeto : arquivosplit
# Arquivo: src/convert_to_utf8.jl    
# Converte arquivos de texto de Windows-1252 (ANSI) para UTF-8
#
# USO:
#   julia convert_to_utf8.jl                        # converte pasta ./TABELAS
#   julia convert_to_utf8.jl /caminho/para/pasta    # converte pasta especificada
#   julia convert_to_utf8.jl arquivo.txt            # converte arquivo único
#   julia convert_to_utf8.jl -h                     # exibe ajuda

const CP1252_MAP = [
    '\u20AC', '\u0081', '\u201A', '\u0192', '\u201E', '\u2026', '\u2020', '\u2021',
    '\u02C6', '\u2030', '\u0160', '\u2039', '\u0152', '\u008D', '\u017D', '\u008F',
    '\u0090', '\u2018', '\u2019', '\u201C', '\u201D', '\u2022', '\u2013', '\u2014',
    '\u02DC', '\u2122', '\u0161', '\u203A', '\u0153', '\u009D', '\u017E', '\u0178'
]

function decode_windows1252(bytes::Vector{UInt8})::String
    chars = Vector{Char}(undef, length(bytes))
    for i in 1:length(bytes)
        b = bytes[i]
        if b <= 0x7F || b >= 0xA0
            chars[i] = Char(b)
        else
            chars[i] = CP1252_MAP[Int(b) - 127]
        end
    end
    return String(chars)
end

function convert_file(file_path::String)::Bool
    if !isfile(file_path)
        @warn "Arquivo não encontrado: $file_path"
        return false
    end

    bytes = read(file_path)

    if isvalid(String, bytes)
        println("  [OK] Já em UTF-8/ASCII: $(basename(file_path))")
        return false
    end

    utf8_str = decode_windows1252(bytes)
    tmp_path = file_path * ".tmp"

    try
        open(tmp_path, "w") do io
            write(io, utf8_str)
        end
        mv(tmp_path, file_path, force=true)
        println("  [CONVERTIDO] $(basename(file_path))")
        return true
    catch e
        isfile(tmp_path) && rm(tmp_path)
        @error "Falha ao converter $(basename(file_path)): $e"
        return false
    end
end

function process_folder(dir::String)
    if !isdir(dir)
        @error "Diretório não encontrado: $dir"
        exit(1)
    end

    println("Diretório: $dir")
    files = filter(isfile, readdir(dir, join=true))

    if isempty(files)
        println("Nenhum arquivo encontrado.")
        return
    end

    converted = 0
    for file in files
        converted += convert_file(file) ? 1 : 0
    end

    println("\nConcluído: $converted de $(length(files)) arquivo(s) convertido(s).")
end

function print_help()
    println("""
    Uso:
      julia convert_to_utf8.jl                        Converte pasta ./TABELAS (padrão)
      julia convert_to_utf8.jl /caminho/para/pasta    Converte todos os arquivos da pasta
      julia convert_to_utf8.jl arquivo.txt            Converte um arquivo específico
      julia convert_to_utf8.jl -h | --help            Exibe esta ajuda
    """)
end

# ── Ponto de entrada ─────────────────────────────────────────────────────────

args = ARGS

if length(args) >= 1 && args[1] in ("-h", "--help")
    print_help()
    exit(0)
end

target = length(args) >= 1 ? args[1] : joinpath(@__DIR__, "TABELAS")

if isfile(target)
    println("Arquivo: $target")
    convert_file(target)
    println("Concluído.")
elseif isdir(target)
    process_folder(target)
else
    @error "Caminho não encontrado: $target"
    print_help()
    exit(1)
end


# julia ./arquivosplit/src/convert_to_utf8.jl