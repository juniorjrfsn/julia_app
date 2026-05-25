# Projeto : arquivosplit
# Arquivo : src/trata_campo_numero.jl
#
# Objetivo: Refatorar campos numéricos em arquivos de dados separados por "|",
#           mantendo apenas dígitos em campos de números inteiros e caracteres
#           decimais válidos em campos decimais.
# ─────────────────────────────────────────────────────────────────────────────

# CREATE TABLE [dbo].[SEGURADO](
# 	[NUM_SEGURADO] [int] NOT NULL,
# 	[COD_BENEF] [char](7) NOT NULL,
# 	[COD_ORGAO] [smallint] NOT NULL,
# 	[SEQ_DEPEND] [smallint] NULL,
# 	[DATA_CADASTRO] [date] NULL,
# 	[UF_ORIGEM] [char](2) NULL,
# 	[NOME] [varchar](50) NOT NULL,
# 	[SEXO] [char](1) NULL,
# 	[CPF] [varchar](11) NULL,
# 	[RG] [varchar](15) NULL,
# 	[ORGAO_EXP_RG] [char](3) NULL,
# 	[UF_EXP_RG] [char](2) NULL,
# 	[DATA_VAL_CARTA] [date] NULL,
# 	[DATA_NASCIMENTO] [date] NULL,
# 	[ESTADO_CIVIL] [char](1) NULL,
# 	[NOME_PAI] [varchar](45) NULL,
# 	[NOME_MAE] [varchar](45) NULL,
# 	[ENDERECO_RESID] [varchar](65) NULL,
# 	[CEP] [char](8) NULL,
# 	[BAIRRO] [varchar](30) NULL,
# 	[COD_MUNICIPIO] [smallint] NULL,
# 	[DATA_NOMEACAO] [date] NULL,
# 	[COD_SIT_ORGAO] [smallint] NULL,
# 	[DATA_SITUACAO] [date] NULL,
# 	[EMITE_CARTEIRA] [char](1) NULL,
# 	[TELEFONE] [varchar](15) NULL,
# 	[USUARIO] [varchar](15) NULL,
# 	[DATA_EMISSAO_CART] [date] NULL,
# 	[COD_MUN_EMISSAO] [int] NULL,
# 	[COD_MUN_ENVIO] [int] NULL,
# 	[EMITIR_OFICIO] [char](1) NULL,
# 	[NUM_OFICIO] [varchar](8) NULL,
# 	[DATA_ATUALIZACAO] [date] NULL,
# 	[COD_CARGO] [varchar](6) NULL,
# 	[VALOR_REMUNERACAO] [decimal](9, 2) NULL,
# 	[FAIXA_ETARIA] [char](6) NULL,
# 	[CLASSE] [varchar](6) NULL,
# 	[REFERENCIA] [varchar](6) NULL,
# 	[TIPO_CARGO] [smallint] NULL,
# 	[OBSERVACAO] [varchar](50) NULL,
# 	[NUM_FOLHA_ORGAO] [varchar](7) NULL,
# 	[COMPL_ENDERECO] [varchar](50) NULL,
# 	[NUM_RESIDENCIA] [smallint] NULL
# ) ON [PRIMARY]

function convert_file(path::String)::Bool
    if !isfile(path)
        @warn "Arquivo não encontrado: $path"
        return false
    end

    println("Lendo arquivo $path...")
    content = read(path, String)
    newline = occursin("\r\n", content) ? "\r\n" : "\n"
    lines = split(content, newline)

    modified = 0
    new_lines = String[]
    sizehint!(new_lines, length(lines))

    # Índices de colunas com tipos numéricos inteiros (1-based):
    # 1: NUM_SEGURADO (int)
    # 3: COD_ORGAO (smallint)
    # 4: SEQ_DEPEND (smallint)
    # 21: COD_MUNICIPIO (smallint)
    # 23: COD_SIT_ORGAO (smallint)
    # 29: COD_MUN_EMISSAO (int)
    # 30: COD_MUN_ENVIO (int)
    # 39: TIPO_CARGO (smallint)
    # 43: NUM_RESIDENCIA (smallint)
    int_indices = [1, 3, 4, 21, 23, 29, 30, 39, 43]

    # Índices de colunas decimais (1-based):
    # 35: VALOR_REMUNERACAO (decimal)
    dec_indices = [35]

    for (idx, line) in enumerate(lines)
        if idx == length(lines) && isempty(line)
            push!(new_lines, "")
            continue
        end

        if idx == 1
            # Cabeçalho
            push!(new_lines, String(line))
            continue
        end

        parts = split(line, '|')
        line_modified = false

        # Trata campos inteiros
        for col_idx in int_indices
            if col_idx <= length(parts)
                val = parts[col_idx]
                if !isempty(val)
                    # Se contiver qualquer caractere não numérico (diferente de dígito)
                    if occursin(r"[^\d]", val)
                        cleaned = filter(isdigit, val)
                        if cleaned != val
                            parts[col_idx] = cleaned
                            line_modified = true
                        end
                    end
                end
            end
        end

        # Trata campos decimais
        for col_idx in dec_indices
            if col_idx <= length(parts)
                val = parts[col_idx]
                if !isempty(val)
                    # Se contiver qualquer caractere que não seja dígito, ponto, vírgula, mais ou menos
                    if occursin(r"[^\d\.,\-\+]", val)
                        cleaned = filter(c -> isdigit(c) || c in ('.', ',', '-', '+'), val)
                        if cleaned != val
                            parts[col_idx] = cleaned
                            line_modified = true
                        end
                    end
                end
            end
        end

        if line_modified
            modified += 1
            push!(new_lines, join(parts, '|'))
        else
            push!(new_lines, String(line))
        end
    end

    if modified == 0
        println("  [ok]        $(basename(path)) (nenhum campo com caracteres não numéricos encontrado)")
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
    println("  [refatorado] $(basename(path)) — $modified linha(s) corrigida(s). Backup em: $backup")
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
        if basename(file) == "SEGURADO.txt"
            converted += convert_file(file) ? 1 : 0
        end
    end

    println("\nConcluído: $converted arquivo(s) refatorado(s).")
end

# ── Ajuda ─────────────────────────────────────────────────────────────────────

function print_help()
    println("""
    Uso:
      julia trata_campo_numero.jl
            Processa o arquivo padrão (SEGURADO.txt)

      julia trata_campo_numero.jl <arquivo>
            Remove caracteres não numéricos dos campos correspondentes no arquivo especificado

      julia trata_campo_numero.jl <diretório>
            Processa o arquivo SEGURADO.txt dentro do diretório especificado

      julia trata_campo_numero.jl -h | --help
            Exibe esta ajuda
    """)
end

# ── Ponto de entrada ──────────────────────────────────────────────────────────

args = ARGS

if length(args) >= 1 && args[1] in ("-h", "--help")
    print_help()
    exit(0)
end

const DEFAULT_TARGET = raw"C:\Users\njunior\Documents\RHFP\TABELA\SEGURADO.txt"
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

# julia ./arquivosplit/src/trata_campo_numero.jl
