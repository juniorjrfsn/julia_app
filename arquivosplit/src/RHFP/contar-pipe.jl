# projeto arquivosplit
# Arquivo: src/contar-pipe.jl


function contar_pipes_por_linha(filepath::String)
    # Gera o nome do arquivo de saída na pasta D:\ contendo a contagem de '|' por linha
    filename = splitext(basename(filepath))[1] * "_pipes_count.txt"
    output_path = joinpath("D:\\", filename)

    total_erros = 0
    max_pipes = 40

    open(output_path, "w") do output
        open(filepath, "r") do file
            for (i, line) in enumerate(eachline(file))
                num_pipes = count(ch -> ch == 'ß', line)
                resultado = "Linha $i: $num_pipes"
                println(resultado)
                println(output, resultado)

                if num_pipes > max_pipes
                    erro = "Linha $i ($num_pipes pipes): $line"
                    println(erro)
                    println(output, "ERRO: " * erro)
                    total_erros += 1
                end
            end
        end
    end

    if total_erros == 0
        println("Nenhuma linha tem mais de $max_pipes pipes.")
    else
        println("\nTotal de linhas com erro: $total_erros")
        println("Resultado salvo em: $output_path")
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) != 1
        println("Uso: julia contar-pipe.jl <caminho_do_arquivo>")
    else
        contar_pipes_por_linha(ARGS[1])
    end
end


# julia .\arquivosplit\src\contar-pipe.jl "D:\projetos\08 - ALAD-ATOS-EVENTOS08 - ALAD-ATOS-EVENTOS.TXT"

 
