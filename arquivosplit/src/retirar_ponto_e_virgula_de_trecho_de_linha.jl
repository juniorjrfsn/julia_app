# projeto arquivosplit
# Arquivo: src/retirar_ponto_e_virgula_de_trecho_de_linha.jl

function refatorar_linhas(filepath::String)
    # Gera o nome do arquivo de saída na pasta D:\AGEPREV-PROJETOS\
    filename = splitext(basename(filepath))[1] * "_linhas_refatoradas.txt"
    output_path = joinpath("D:\\AGEPREV-PROJETOS", filename)
    
    open(output_path, "w") do output
        open(filepath, "r") do file
            for (i, line) in enumerate(eachline(file))
                
                # Replace ponto e virgula por dois pontos em trechos de linha
                # Campos : [ALA-AE-NUMERO-PROCESSO], [INSTRUMENTO-LEGAL], [ALA-AE-HISTORICO], [ALA-AE-CARGO-MT1]

                if length(line) >= 106 
                    campo1_inicio = 106 # ✅ [ALA-AE-NUMERO-PROCESSO]
                    campo1_fim = min(117, length(line))
                    trecho1 = replace(line[campo1_inicio:campo1_fim], ";" => "|")

                    campo2_inicio = 119  # ✅ [INSTRUMENTO-LEGAL]
                    campo2_fim = min(268, length(line))
                    trecho2 = replace(line[campo2_inicio:campo2_fim], ";" => "|")

                    campo3_inicio = 354   # ✅ [ALA-AE-HISTORICO]
                    campo3_fim = min(853, length(line)) 
                    trecho3 = replace(line[campo3_inicio:campo3_fim], ";" => "|")

                    campo4_inicio = 1225   # ✅ [ALA-AE-CARGO-MT1]
                    campo4_fim = min(1234, length(line))
                    trecho4 = replace(line[campo4_inicio:campo4_fim], ";" => "|")

                    nova_linha = 
                    line[1:campo1_inicio-1]            * trecho1 *   # ✅ [ALA-AE-NUMERO-PROCESSO]
                    line[campo1_fim+1:campo2_inicio-1] * trecho2 *   # ✅ [INSTRUMENTO-LEGAL]
                    line[campo2_fim+1:campo3_inicio-1] * trecho3 *   # ✅ [ALA-AE-HISTORICO]
                    line[campo3_fim+1:campo4_inicio-1] * trecho4 *   # ✅ [ALA-AE-CARGO-MT1]
                    line[campo4_fim+1:end]

                    println(output, nova_linha)
                else
                    println(output, line)
                end
            end
        end
    end

    println("\nResultado salvo em: $output_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) != 1
        println("Uso: julia contar_ponto_e_viegula_por_linha.jl <caminho_do_arquivo>")
    else
        refatorar_linhas(ARGS[1])
    end
end


# julia .\arquivosplit\src\retirar_ponto_e_virgula_de_trecho_de_linha.jl "D:\AGEPREV-PROJETOS\ALY.S.BIFPATEV.d17072017_linhas_truncadas.txt"
# julia .\arquivosplit\src\retirar_ponto_e_virgula_de_trecho_de_linha.jl "D:\AGEPREV-PROJETOS\truncadas.txt"
# julia .\arquivosplit\src\retirar_ponto_e_virgula_de_trecho_de_linha.jl "D:\AGEPREV-PROJETOS\ALY.S.BIFPATEV.d17072017.txt"

