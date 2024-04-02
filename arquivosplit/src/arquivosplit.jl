# read file contents, one line at a time
 
open("ARQUIVO 2 PESSOAS  2024-03 - v.1.0.csv") do f
 
    # line_number
    line = 0  
    linhas = 0
    # read till end of file
    while ! eof(f) 
   
       # read a new / next line for every iteration          
       s = readline(f)         
       line += 1
       println("$line . $s")
       linhas += 1
    end
    println("$linhas . $linhas")
  end


  # julia arquivosplit.jl
