# projeto arquivosplit
# Arquivo: src/trocar-pipe-por-beta.jl


function trocar_pipe_campo8_por_beta(filepath::String)
    # Gera o nome do arquivo de saída na pasta D:\projetos\
    filename = splitext(basename(filepath))[1] * "08 - ALAD-ATOS-EVENTOS.TXT"
    output_path = joinpath("D:\\projetos\\", filename)

    modified_lines = 0
    cnt = 0
    positions = (8, 14, 20
,29
,32
,41
,50
,59
,65
,75
,85
,91
,97
,110
,351
,862
,866
,871
,962
,1023
,1154
,1168
,1182
,1186
,1189
,1200
,1217
,1219
,1233
,1238
,1240
,1242
,1245
,1247
,1249
,1251
,1262
,1273
,1282)

    open(output_path, "w") do output
        open(filepath, "r") do file
            for line in eachline(file)
                chars = collect(line)
                modified = false

                if cnt == 0
                    # Primeira linha (cabeçalho): substituir todos os pipes por ß
                    line = replace(line, '|' => 'ß')
                    modified_lines += 1
                else
                    # Demais linhas: substituir apenas nas posições específicas
                    for pos in positions
                        if pos <= length(chars) && chars[pos] == '|'
                            chars[pos] = 'ß'
                            modified = true
                        end
                    end
                    if modified
                        modified_lines += 1
                    end
                    line = String(chars)
                end

                #println(line)
                println(output, line)
                cnt += 1
            end
        end
    end

    println("Linhas modificadas: $modified_lines")
    println("Resultado salvo em: $output_path")
end


if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) != 1
        println("Uso: julia trocar-pipe-por-beta.jl <caminho_do_arquivo>")
    else
        trocar_pipe_campo8_por_beta(ARGS[1])
    end
end


# julia .\arquivosplit\src\trocar-pipe-por-beta.jl "D:\projetos\julia_app\arquivosplit\src\08 - ALAD-ATOS-EVENTOS.TXT"
