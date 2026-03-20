function contar_caracteres_por_linha(filepath::String)
    # Gera o nome do arquivo de saída na pasta D:\AGEPREV-PROJETOS\
    filename = splitext(basename(filepath))[1] * "_linhas_fora_do_padrao.txt"
    output_path = joinpath("D:\\AGEPREV-PROJETOS", filename)

    total_erros = 0

    open(output_path, "w") do output
        open(filepath, "r") do file
            for (i, line) in enumerate(eachline(file))
                num_chars = length(line)
                if num_chars > 1261
                    resultado = "Linha $i ($num_chars caracteres): $line"
                    println(resultado)
                    println(output, resultado)
                    total_erros += 1
                end
                if num_chars < 1261
                    resultado = "Linha $i ($num_chars caracteres): $line"
                    println(resultado)
                    println(output, resultado)
                    total_erros += 1
                end
            end
        end
    end

    if total_erros == 0
        println("Nenhuma linha é menor que 1261 ou maior que 1261 caracteres.")
    else
        println("\nTotal de linhas com erro: $total_erros")
        println("Resultado salvo em: $output_path")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) != 1
        println("Uso: julia contar_caracteres_por_linha.jl <caminho_do_arquivo>")
    else
        contar_caracteres_por_linha(ARGS[1])
    end
end


# julia .\arquivosplit\src\contar_caracteres_por_linha.jl "D:\AGEPREV-PROJETOS\ALY.S.BIFPATEV.d17072017.txt"

# julia .\arquivosplit\src\contar_caracteres_por_linha.jl "D:\AGEPREV-PROJETOS\ALY.S.BIFPATEV.d17072017_linhas_truncadas_linhas_refatoradas.txt"