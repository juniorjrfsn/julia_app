# projeto arquivosplit
# Arquivo: src/contar_ponto_e_viegula_por_linha.jl

function contar_ponto_e_virgula(filepath::String)
    # Gera o nome do arquivo de saída na pasta D:\AGEPREV-PROJETOS\
    filename = splitext(basename(filepath))[1] * "_linhas_truncadas.txt"
    output_path = joinpath("D:\\AGEPREV-PROJETOS", filename)
    
    open(output_path, "w") do output
        open(filepath, "r") do file
            for (i, line) in enumerate(eachline(file))
                num_semicolons = count(==(';'), line)
                if num_semicolons > 63
                    resultado = "Linha $i ($num_semicolons ponto e vírgulas): $line"
                    println(resultado)
                    println(output, line)
                end
                if num_semicolons < 63
                    resultado = "Linha $i ($num_semicolons ponto e vírgulas): $line"
                    println(resultado)
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
        contar_ponto_e_virgula(ARGS[1])
    end
end

# julia .\arquivosplit\src\contar_ponto_e_viegula_por_linha.jl "D:\AGEPREV-PROJETOS\ALY.S.BIFPATEV.d17072017.txt"

# julia .\arquivosplit\src\contar_ponto_e_viegula_por_linha.jl "D:\AGEPREV-PROJETOS\ALY.S.BIFPATEV.d17072017_linhas_truncadas_linhas_refatoradas.txt"

# julia .\arquivosplit\src\contar_ponto_e_viegula_por_linha.jl "D:\AGEPREV-PROJETOS\ALY.S.BIFPATEV.d17072017_linhas_refatoradas.txt"
