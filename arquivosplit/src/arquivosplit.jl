# projeto arquivosplit
# Arquivo: src/arquivosplit.jl

module arquivosplit

    using FileIO
    using TOML

    try
        # -----------------------------------------------------------------
        # Resolução do diretório de trabalho:
        # 1) Se for passado um argumento na linha de comando, usa ele.
        # 2) Caso contrário, usa o diretório onde este script está localizado
        #    (funciona independente de onde o `julia` foi chamado).
        # -----------------------------------------------------------------
        diretorio = if !isempty(ARGS)
            ARGS[1]
        else
            @__DIR__
        end

        # normaliza (remove barra final duplicada, resolve caminho relativo, etc.)
        diretorio = normpath(diretorio)

        caminho_toml = joinpath(diretorio, "arquivosplit.toml")

        if !isfile(caminho_toml)
            error("Arquivo de configuração não encontrado: $caminho_toml")
        end

        # Diretório base para resolver os caminhos relativos DENTRO do .toml
        # (arquivo_nomeorigem / arquivo_nome_parte). Usa o MESMO diretório
        # do script/config (diretorio), e não o diretório atual (pwd()) —
        # assim o resultado é sempre o mesmo, não importa de onde ou como
        # o script é chamado (julia direto, .bat com cd, duplo-clique etc).
        #
        # Por isso, os caminhos no .toml devem ser relativos a essa pasta,
        # ex.: arquivo_nomeorigem = "matriz.jl"  (não "./arquivosplit/src/matriz.jl")
        base_dir = diretorio

        println("\033[1;34mDiretório do script/config/base: \033[1;32m$diretorio")
        println("\033[1;34mArquivo de configuração: \033[1;32m$caminho_toml\n")

        arquivosplit_config = TOML.parsefile(caminho_toml)
        dados = arquivosplit_config["dados"]

        for arquivo_info in dados["arquivo"]
            arquivo_nomeorigem  = arquivo_info["arquivo_nomeorigem"]
            arquivo_qtde_parte  = arquivo_info["arquivo_qtde_parte"]
            arquivo_nome_parte  = arquivo_info["arquivo_nome_parte"]

            # Se o nome informado no TOML já for um caminho absoluto, joinpath
            # simplesmente devolve esse caminho; se for relativo, resolve
            # em relação ao diretório atual (base_dir).
            arquivo          = joinpath(base_dir, arquivo_nomeorigem)
            caminho_saida     = joinpath(base_dir, arquivo_nome_parte)
            num_arq           = arquivo_qtde_parte
            num_lin_por_arq   = 0

            if !isfile(arquivo)
                println("\033[1;31mArquivo de origem não encontrado: $arquivo\n")
                continue
            end

            try
                # Primeira passada: conta o total de linhas do arquivo
                open(arquivo) do f
                    lin = 0
                    while !eof(f)
                        readline(f)
                        lin += 1
                    end
                    println("")

                    num_lin_por_arq = trunc(Int, (lin / num_arq))
                    println("\033[1;34mQtde de arquivos: \033[1;32m$num_arq")
                    println("\033[1;34mnumero de linhas por arquivo: \033[1;32m$num_lin_por_arq")
                    println("\033[1;34mTotal de linhas: \033[1;32m$lin")
                    println("\033[1;33mProcessando ...\n")

                    try
                        # Segunda passada: efetivamente divide o arquivo em partes
                        open(arquivo, "r") do f2
                            cnt     = 0
                            lnhs    = 0
                            n_arqs  = 1
                            linha   = ""

                            for line in eachline(f2)
                                cnt  += 1
                                lnhs += 1

                                if lnhs == 1
                                    println("\033[1;34marquivo: \033[1;32m$n_arqs")
                                end

                                if lnhs == num_lin_por_arq
                                    if n_arqs == num_arq
                                        # última parte: acumula até o fim do arquivo
                                        if cnt == lin
                                            linha = "$linha$line"
                                        else
                                            linha = "$linha$line\n"
                                        end
                                    else
                                        println("\n")
                                        linha = "$linha$line"
                                        f_arquivo_1 = open("$caminho_saida$n_arqs.csv", "w")
                                        write(f_arquivo_1, "$linha")
                                        close(f_arquivo_1)
                                        lnhs  = 0
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

                                if lnhs != 0
                                    print("\r\033[1;34mQtde linhas: \033[1;32m$lnhs \033[1;34mtotal de linhas processadas do arquivo fonte: \033[1;32m$cnt")
                                end
                            end

                            println("\n")
                            f_arquivo = open("$caminho_saida$num_arq.csv", "w")
                            write(f_arquivo, "$linha")
                            close(f_arquivo)
                            close(f2)
                        end
                    catch e
                        println("Erro ao processar o arquivo: $(e)")
                    end
                end
            catch e
                println("Erro ao abrir o arquivo: $(e)")
            end

            println("\033[1;33mProcesso executado com sucesso!\e[1;30m\n")
        end
    catch e
        println("Erro: $(e)")
    end

    # Execução (usando o diretório do próprio script):
    #   julia arquivosplit/src/arquivosplit.jl
    # Execução (informando outro diretório onde estão o .toml e o arquivo fonte):
    #   julia arquivosplit/src/arquivosplit.jl arquivosplit/src
    # Execução: ./arquivosplit/src/dividir.bat
	# trunc(2.25)   = 2
	# floor(2.25)   = 2
	# round(2.25)   = 2
	# Int(2.25)     = 2
end # module arquivosplit
