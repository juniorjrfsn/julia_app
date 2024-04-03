# read file contents, one line at a time
using FileIO 
#import Pkg;
#Pkg.add("IO")
arquivo = "ARQUIVO 2 PESSOAS  2024-03 - v.1.0.csv"
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
      println("linhas = $lin")
      num_lin_por_arq = trunc(Int,(lin / num_arq))
      println("Qtde de arquivos = $num_arq")
      println("numero de linhas por arquivo = $num_lin_por_arq")
      println("Total de linhas = $lin")
      println("Processando ...")
      try
         open(arquivo, "r") do f
            cnt = 0
            lnhs = 0
            n_arqs = 0
            linha = ""
            #seekstart(f)
            for line in eachline(f) 
               #println(line)
               cnt += 1
               lnhs += 1
               #println("posi = $lnhs")
               if lnhs == num_lin_por_arq
                  n_arqs += 1
                  
                  if n_arqs == num_arq
                     # code to execute if the condition is true
                     if cnt == lin
                        linha = "$linha$line"
                     else
                        linha = "$linha$line\n"
                     end 
                  else
                     linha = "$linha$line"
                     println("arquivo = $n_arqs")
                     println("Qtde linhas = $lnhs")
                     arquivo = open("arquivo_$n_arqs.csv", "w")
                     write(arquivo, "$linha")
                     close(arquivo)
                     lnhs = 0
                     linha = ""
                   end
               else
                  if cnt == lin
                     linha = "$linha$line"
                  else
                     linha = "$linha$line\n"
                  end 
               end

            end
            println("arquivo = $n_arqs")
            println("Qtde linhas = $lnhs")
            arquivo = open("arquivo_$n_arqs.csv", "w")
            write(arquivo, "$linha")
            close(arquivo)
            println("arquivo = $n_arqs")
            close(f) 
         end
      catch e
         println("Erro ao abrir o arquivo: $(e.message)")
      end
   end
catch e
   println("Erro ao abrir o arquivo: $(e.message)")
end
println("Processo executado com sucesso!")



# julia arquivosplit.jl
#trunc(2.25) = 2
#floor(2.25) = 2
#round(2.25) = 2
#Int(2.25) = 2