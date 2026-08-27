using JSON

function generate_header(input_file, output_file)
    println("Lendo arquivo: $input_file")
    
    # Read the JSON file and handle potential BOM
    content = read(input_file, String)
    if startswith(content, "\ufeff")
        content = lstrip(content, '\ufeff')
    end
    data = JSON.parse(content)
    
    # Get unique table names to define the order of generation
    table_names = unique([item["NOME_DA_TABELA"] for item in data])
    
    println("Encontradas $(length(table_names)) tabelas.")
    
    output_data = []
    
    for table_name in table_names
        # Filtra os dados da tabela atual
        columns = filter(item -> item["NOME_DA_TABELA"] == table_name, data)
        
        # Extrai apenas os nomes das colunas
        col_names = [col["NOME_DA_COLUNA"] for col in columns]
        
        # Concatena os nomes separados por '|'
        header_str = join(col_names, "|")
        
        # Adiciona ao array de saída
        push!(output_data, Dict("NOME_DA_TABELA" => table_name, "HEADER" => header_str))
    end
    
    open(output_file, "w") do f
        JSON.print(f, output_data, 4)
    end
    
    println("Arquivo gerado com sucesso em: $output_file")
end

# Execução para o schema MSPREV
input_json = "C:/Users/njunior/Documents/RHFP/MSPREV_SCHEMA_HEADER.json"
output_json = "C:/Users/njunior/Documents/RHFP/MSPREV_HEADERS.json"

if isfile(input_json)
    try
        generate_header(input_json, output_json)
    catch e
        println("Erro durante o processamento de $input_json: $e")
    end
else
    println("Erro: Arquivo '$input_json' não encontrado.")
end

# Execução para o schema PREVI
input_json_2 = "C:/Users/njunior/Documents/RHFP/PREVI_SCHEMA_HEADER.json"
output_json_2 = "C:/Users/njunior/Documents/RHFP/PREVI_HEADERS.json"

if isfile(input_json_2)
    try
        generate_header(input_json_2, output_json_2)
    catch e
        println("Erro durante o processamento de $input_json_2: $e")
    end
else
    println("Erro: Arquivo '$input_json_2' não encontrado.")
end


# julia arquivosplit/src/gera_header.jl
