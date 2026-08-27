function gerar_comandos(tabela_file, saida_file)
    # Verifica se o arquivo de entrada existe
    if !isfile(input_file)
        println("Erro: O arquivo $input_file não foi encontrado.")
        return
    end

    open(input_file, "r") do input
        open(output_file, "w") do output
            for line in eachline(input)
                tabela = strip(line)
                if !isempty(tabela)
                    # O usuário pediu especificamente este formato:
                    # EXPORT TO C:\DB2_BKP\TTTTAAAABBEELLLAAAAA.txt OF DEL MODIFIED BY COLDEL| NOCHARDEL DATESISO MESSAGES C:\DB2_BKP\TTTTAAAABBEELLLAAAAA.msg SELECT * FROM UDB2.TTTTAAAABBEELLLAAAAA;
                    
                    cmd = "EXPORT TO C:\\DB2_BKP\\$tabela.txt OF DEL MODIFIED BY COLDEL| NOCHARDEL DATESISO MESSAGES C:\\DB2_BKP\\$tabela.msg SELECT * FROM UDB2.$tabela;"
                    println(output, cmd)
                end
            end
        end
    end
    println("Arquivo $output_file gerado com sucesso!")
end

# Executa a função
input_file = "C:/Users/njunior/Documents/RHFP/tabela_MSPREV.txt"
output_file = "C:/Users/njunior/Documents/RHFP/comandos_db2_MSPREV.txt"

gerar_comandos(input_file, output_file)

input_file = "C:/Users/njunior/Documents/RHFP/tabela_PREVI.txt"
output_file = "C:/Users/njunior/Documents/RHFP/comandos_db2_PREVI.txt"

gerar_comandos(input_file, output_file)


# julia arquivosplit/src/gera_arquivo.jl
