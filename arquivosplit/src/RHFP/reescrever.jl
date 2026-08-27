# Projeto : arquivosplit
# Arquivo: src/reescrever.jl
""" 
Lê um arquivo de texto linha por linha, onde cada linha tem campos separados por `;`.
Salva as informações em um novo arquivo, encapsulando cada valor em aspas duplas.
"""
function reescrever_arquivo(caminho_entrada::String, caminho_saida::String)
    # Abre o arquivo de saída para escrita e o de entrada para leitura
    open(caminho_saida, "w") do out
        open(caminho_entrada, "r") do in_file
            # Função auxiliar para verificar se o campo tem apenas regras de espaço ou "null"
            is_nulo(c) = isempty(strip(replace(lowercase(c), "null" => "")))
            
            # Função para processar e limpar o campo
            function limpa_campo(c)
                is_nulo(c) && return "0"
                # Transforma espaços em '0'
                c = replace(c, " " => "0")
                # Remove qualquer caracter que NÃO seja a-z, A-Z, 0-9, ou +, -, ., ,
                return replace(c, r"[^a-zA-Z0-9\+\-\.,]" => "")
            end

            for linha in eachline(in_file)
                # Separa os campos pelo separador ";"
                campos = split(linha, ";")
                
                # Aplica as limpezas de caracteres e tratamento em todos os campos
                campos_tratados = limpa_campo.(campos)
                
                # Encapsula cada campo adicionando aspas duplas no início e no fim
                campos_com_aspas = ["\"$campo\"" for campo in campos_tratados]
                
                # Remonta a linha usando o ";" como separador
                nova_linha = join(campos_com_aspas, ";")
                
                # Grava a linha formatada no arquivo de saída
                println(out, nova_linha)
            end
        end
    end
    println("Processo concluído. Arquivo salvo em: ", caminho_saida)
end

# Exemplo de uso:
# reescrever_arquivo("entrada.txt", "saida.txt")

reescrever_arquivo("C:\\Users\\njunior\\Documents\\projetos\\ALY.S.BIFPFIA3.d27072017_.txt",
 "C:\\Users\\njunior\\Documents\\projetos\\ALY.S.BIFPFIA3.d27072017.txt")

# julia arquivosplit/src/reescrever.jl
