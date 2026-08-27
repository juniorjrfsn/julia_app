# Projeto : arquivosplit
# Arquivo : src/retirar_quebra_cr.jl
#
# Objetivo: Remover CR (\r, 0x0D) soltos de arquivos de dados cujas quebras
#           legítimas de registro são CRLF (\r\n).
#
# Estratégia (operação em bytes, sem decodificar texto):
#   Percorre o buffer byte a byte; ao encontrar 0x0D verifica se o próximo
#   byte é 0x0A — se sim, mantém ambos (quebra de registro legítima);
#   caso contrário, descarta o 0x0D (CR solto dentro de campo).
# ─────────────────────────────────────────────────────────────────────────────

"""
    remove_standalone_cr(data::Vector{UInt8}) -> Vector{UInt8}

Retorna uma cópia de `data` sem os bytes CR (0x0D) que não são seguidos de LF (0x0A).
Os pares CRLF são preservados intactos.
"""
function remove_standalone_cr(data::Vector{UInt8})::Vector{UInt8}
    n = length(data)
    out = UInt8[]
    sizehint!(out, n)          # evita realocações excessivas

    i = 1
    while i <= n
        b = data[i]
        if b == 0x0D
            if i < n && data[i + 1] == 0x0A
                # CRLF legítimo → mantém os dois bytes
                push!(out, 0x0D, 0x0A)
                i += 2
            else
                # CR solto → descarta
                i += 1
            end
        else
            push!(out, b)
            i += 1
        end
    end

    return out
end

"""
    convert_file(path::String) -> Bool

Lê o arquivo em `path`, remove CRs soltos e sobrescreve o arquivo.
Retorna `true` se o arquivo foi modificado, `false` se já estava correto.
Cria backup `.bak` antes de qualquer modificação.
"""
function convert_file(path::String)::Bool
    original = read(path)              # lê bytes brutos
    fixed    = remove_standalone_cr(original)

    if fixed == original
        println("  [ok]      $path  (sem alterações necessárias)")
        return false
    end

    removed = length(original) - length(fixed)

    # Backup antes de sobrescrever
    backup = path * ".bak"
    cp(path, backup; force=true)

    write(path, fixed)
    println("  [corrigido] $path  ($removed CR(s) solto(s) removido(s)) — backup: $backup")
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
        converted += convert_file(file) ? 1 : 0
    end

    println("\nConcluído: $converted de $(length(files)) arquivo(s) corrigido(s).")
end

# ── Ajuda ─────────────────────────────────────────────────────────────────────

function print_help()
    println("""
    Uso:
      julia retirar_quebra_cr.jl
            Processa o arquivo padrão (HIST_PESSOAL.txt)

      julia retirar_quebra_cr.jl <arquivo>
            Remove CR soltos de um arquivo específico

      julia retirar_quebra_cr.jl <diretório>
            Remove CR soltos de todos os arquivos na pasta

      julia retirar_quebra_cr.jl -h | --help
            Exibe esta ajuda

    Comportamento:
      • Preserva CRLF (\\r\\n) — quebras legítimas de registro
      • Remove CR (\\r) soltos encontrados dentro de campos
      • Cria backup <arquivo>.bak antes de modificar
    """)
end

# ── Ponto de entrada ──────────────────────────────────────────────────────────

args = ARGS

if length(args) >= 1 && args[1] in ("-h", "--help")
    print_help()
    exit(0)
end

const DEFAULT_TARGET = raw"C:\Users\njunior\Documents\RHFP\TABELA\HIST_PESSOAL.txt"
target = length(args) >= 1 ? args[1] : DEFAULT_TARGET

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

# julia ./arquivosplit/src/retirar_quebra_cr.jl
