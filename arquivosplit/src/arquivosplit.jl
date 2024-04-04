module arquivosplit
	using FileIO
   using TOML

	try
		#io = open("arquivosplit.toml", "r");
      #println(read(io, String))
      #println(TOML.parsefile("arquivosplit.toml"))
      arquivosplit = TOML.parsefile("arquivosplit.toml")
      dados = arquivosplit["dados"]
      for arquivo_info in dados["arquivo"]
         arquivo_nomeorigem = arquivo_info["arquivo_nomeorigem"]
         arquivo_qtde_parte = arquivo_info["arquivo_qtde_parte"]
         arquivo_nome_parte = arquivo_info["arquivo_nome_parte"]
         #println(arquivo_nomeorigem)
         #println(arquivo_qtde_parte)
         #println(arquivo_nome_parte)
         #dados_arquivo = arquivo_info["dados"]
         #tipo_arquivo = dados_arquivo["tipo"]
         #tamanho_arquivo = dados_arquivo["tamanho"]
         #println(tipo_arquivo)
         #println(tamanho_arquivo)

 

         arquivo = arquivo_nomeorigem
         num_arq = arquivo_qtde_parte
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
                              f_arquivo_1 = open("$arquivo_nome_parte$n_arqs.csv", "w")
                              write(f_arquivo_1, "$linha")
                              close(f_arquivo_1)
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
                     f_arquivo = open("$arquivo_nome_parte$num_arq.csv", "w")
                     write(f_arquivo, "$linha")
                     close(f_arquivo)
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
      










      end
	catch e
		println("Erro ao abrir o arquivo: $(e.message)")
	end


	#  julia arquivosplit.jl

	#  trunc(2.25)   = 2
	#  floor(2.25)   = 2
	#  round(2.25)   = 2
	#  Int(2.25)     = 2
end # module arquivosplit