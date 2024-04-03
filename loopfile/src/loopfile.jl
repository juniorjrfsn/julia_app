module loopfile
    arquivo = "deps.csv"
    num_arq = 4
    num_lin_por_arq = 0
    try
    open(arquivo) do f
        # line_number
        lin = 0
        # read till end of file
        while ! eof(f)
            # read a new / next line for every iteration          
            s = readline(f)
            lin += 1
            #println("$line . $s")
        end
        println("")
   
        num_lin_por_arq = trunc(Int,(lin / num_arq))
        println("\033[1;34mQtde de arquivos: \033[1;32m$num_arq")
        println("\033[1;34mnumero de linhas por arquivo: \033[1;32m$num_lin_por_arq")
        println("\033[1;34mTotal de linhas: \033[1;32m$lin")
        println("\033[1;33mProcessando ...\n")
        try
            open(arquivo, "r") do f
                cnt = 0
                lnhs = 0
                n_arqs = 1
                linha = ""
                #seekstart(f)
                for line in eachline(f) 
                    #println(line)
                    cnt += 1
                    lnhs += 1
                    #println("posi = $lnhs")
                    if lnhs == 1
                        println("\033[1;34marquivo: \033[1;32m$n_arqs")
                    end
                    if lnhs == num_lin_por_arq
                        
                        if n_arqs == num_arq
                            # code to execute if the condition is true
                            if cnt == lin
                                linha = "$linha$line"
                            else
                                linha = "$linha$line\n"
                            end 
                        else
                            println("\n")
                            linha = "$linha$line"
                            arquivo = open("arquivo_$n_arqs.csv", "w")
                            write(arquivo, "$linha")
                            close(arquivo)
                            lnhs = 0
                            linha = ""
                        end
                        n_arqs += 1
                    else
                        if cnt == lin
                            linha = "$linha$line"
                        else
                            linha = "$linha$line\n"
                        end 
                    end
                    if lnhs == 0
                    else
                        print("\r\033[1;34mQtde linhas: \033[1;32m$lnhs \033[1;34mtotal de linhas processadas do arquivo fonte: \033[1;32m$cnt")
                    end
                end
                println("\n")
                arquivo = open("arquivo_$n_arqs.csv", "w")
                write(arquivo, "$linha")
                close(arquivo)
                close(f)
            end
        catch e
            println("Erro ao abrir o arquivo: $(e.message)")
        end
    end
    catch e
    println("Erro ao abrir o arquivo: $(e.message)")
    end
    println("\033[1;33mProcesso executado com sucesso!\e[1;30m\n")


    # julia loopfile.jl
    # trunc(2.25) = 2
    # floor(2.25) = 2
    # round(2.25) = 2
    # Int(2.25) = 2


end # module loopfile
