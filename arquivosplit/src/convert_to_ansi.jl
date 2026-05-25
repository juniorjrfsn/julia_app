#!/usr/bin/env julia
# Projeto : arquivosplit
# Arquivo: src/convert_to_ansi.jl
# Converte arquivos de texto de UTF-8 para Windows-1252 (ANSI)
#
# USO:
#   julia convert_to_ansi.jl                        # converte pasta ./TABELAS
#   julia convert_to_ansi.jl /caminho/para/pasta    # converte pasta especificada
#   julia convert_to_ansi.jl arquivo.txt            # converte arquivo único
#   julia convert_to_ansi.jl -h                     # exibe ajuda

# Mapeamento do Windows-1252 para a faixa de bytes 0x80 a 0x9F (128 a 159)
const CP1252_MAP = [
    '\u20AC', '\u0081', '\u201A', '\u0192', '\u201E', '\u2026', '\u2020', '\u2021',
    '\u02C6', '\u2030', '\u0160', '\u2039', '\u0152', '\u008D', '\u017D', '\u008F',
    '\u0090', '\u2018', '\u2019', '\u201C', '\u201D', '\u2022', '\u2013', '\u2014',
    '\u02DC', '\u2122', '\u0161', '\u203A', '\u0153', '\u009D', '\u017E', '\u0178'
]

# Tabela de busca reversa para mapear caracteres UTF-8 de volta para bytes Windows-1252
const UTF8_TO_CP1252 = Dict{Char, UInt8}()
for (i, char) in enumerate(CP1252_MAP)
    UTF8_TO_CP1252[char] = UInt8(i + 127)
end

# Converte uma string UTF-8 do Julia para uma sequência de bytes Windows-1252 (ANSI)
# Usa IOBuffer para evitar pré-alocação incorreta com length() em strings multibyte
function encode_windows1252(s::String)::Vector{UInt8}
    buf = IOBuffer()
    for c in s
        cp = Int(c)
        if cp <= 0x7F
            write(buf, UInt8(cp))
        elseif 0xA0 <= cp <= 0xFF
            write(buf, UInt8(cp))
        elseif haskey(UTF8_TO_CP1252, c)
            write(buf, UTF8_TO_CP1252[c])
        else
            @warn "Caractere U+$(uppercase(string(cp, base=16, pad=4))) não mapeável em CP1252, substituído por '?'"
            write(buf, UInt8('?'))
        end
    end
    return take!(buf)
end

# Converte um único arquivo de UTF-8 para ANSI
function convert_file(file_path::String)::Bool
    if !isfile(file_path)
        @warn "Arquivo não encontrado: $file_path"
        return false
    end

    bytes = read(file_path)

    # Se os bytes já NÃO formarem uma string UTF-8 válida, assume-se que o arquivo já está em ANSI
    if !isvalid(String, bytes)
        println("  [OK] Já em ANSI (não é UTF-8 válido): $(basename(file_path))")
        return false
    end

    # Se for UTF-8 válido mas contiver apenas caracteres ASCII (faixa 0x00-0x7F),
    # ele já é compatível e idêntico em ANSI, dispensando conversão
    if all(b -> b <= 0x7F, bytes)
        println("  [OK] Já em ANSI (ASCII puro): $(basename(file_path))")
        return false
    end

    utf8_str = String(bytes)
    ansi_bytes = encode_windows1252(utf8_str)
    tmp_path = file_path * ".tmp"

    try
        # Abre em modo binário explícito (write=true) para evitar
        # conversão automática de line endings no Windows (\n → \r\n)
        open(tmp_path, write=true) do io
            write(io, ansi_bytes)
        end
        mv(tmp_path, file_path, force=true)
        println("  [CONVERTIDO PARA ANSI] $(basename(file_path))")
        return true
    catch e
        isfile(tmp_path) && rm(tmp_path)
        @error "Falha ao converter $(basename(file_path)) para ANSI: $e"
        return false
    end
end

# Processa todos os arquivos de um diretório
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
    skipped   = 0
    for file in files
        # Ignora arquivos de script do próprio projeto
        if endswith(file, ".jl") || endswith(file, ".bat") || endswith(file, ".toml")
            skipped += 1
            continue
        end
        converted += convert_file(file) ? 1 : 0
    end

    total = length(files) - skipped
    println("\nConcluído: $converted de $total arquivo(s) convertido(s) para ANSI.")
end

# Exibe as instruções de uso
function print_help()
    println("""
    Uso:
      julia convert_to_ansi.jl                        Converte pasta ./TABELAS (padrão)
      julia convert_to_ansi.jl /caminho/para/pasta    Converte todos os arquivos da pasta
      julia convert_to_ansi.jl arquivo.txt            Converte um arquivo específico
      julia convert_to_ansi.jl -h | --help            Exibe esta ajuda
    """)
end

# ── Ponto de entrada (Main) ──────────────────────────────────────────────────

args = ARGS

if length(args) >= 1 && args[1] in ("-h", "--help")
    print_help()
    exit(0)
end

# Define a pasta padrão (TABELAS) ou o argumento informado
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


# julia ./arquivosplit/src/convert_to_ansi.jl