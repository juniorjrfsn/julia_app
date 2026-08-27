using JSON

function generate_ddl(input_file, output_file)
    println("Lendo arquivo: $input_file")
    
    # Read the JSON file and handle potential BOM
    content = read(input_file, String)
    if startswith(content, "\ufeff")
        content = lstrip(content, '\ufeff')
    end
    data = JSON.parse(content)
    
    # "fazer uma varredura somente pelo nome da tabela"
    # Get unique table names to define the order of generation
    table_names = unique([item["NOME_DA_TABELA"] for item in data])
    
    println("Encontradas $(length(table_names)) tabelas.")
    
    open(output_file, "w") do f
        for table_name in table_names
            # "ler novamente o arquivo e buscar como chave o nome NOME_DA_TABELA"
            # Here we filter the already loaded data for the current table
            columns = filter(item -> item["NOME_DA_TABELA"] == table_name, data)
            
            write(f, "CREATE TABLE $table_name (\n")
            
            for (i, col) in enumerate(columns)
                col_name = col["NOME_DA_COLUNA"]
                data_type = col["TIPO_DE_DADO"]
                size = col["TAMANHO"]
                decimals = col["DECIMAIS"]
                nullable = col["ACEITA_NULO"]
                
                # Logic for type formatting:
                # "adicionando NOME_DA_COLUNA, TIPO_DE_DADO, TAMANHO, DECIMAIS caso tiver senão ignorar"
                type_str = data_type
                if decimals > 0
                    type_str = "$data_type($size, $decimals)"
                elseif size > 0 && !(data_type in ["DATE", "INTEGER", "SMALLINT", "TIMESTAMP"])
                    type_str = "$data_type($size)"
                end

                if uppercase(strip(type_str)) == "LONG VARCHAR(32700)"
                    type_str = "TEXT"
                elseif uppercase(strip(type_str)) == "BIGINT(8)"
                    type_str = "INTEGER"
                elseif uppercase(strip(type_str)) == "DOUBLE(8)"
                    type_str = "NUMERIC(15,2)"
                elseif uppercase(strip(type_str)) == "CLOB(1048576)"
                    type_str = "TEXT"
                elseif uppercase(strip(type_str)) == "BLOB(10485760)"
                    type_str = "TEXT"
                elseif uppercase(strip(type_str)) == "BLOB(16777216)"
                    type_str = "TEXT"
                end
                
                if occursin("VARCHAR", uppercase(type_str)) || occursin("TEXT", uppercase(type_str))
                    type_str *= " COLLATE Latin1_General_100_CI_AS"
                end
                
                # "acrescentando NULL como default"
                # If ACEITA_NULO is 'Y', we add NULL. Otherwise NOT NULL.
                null_str = (nullable == "Y") ? "NULL" : "NOT NULL"
                
                line = "    $col_name $type_str $null_str"
                
                # Add comma if not the last column
                if i < length(columns)
                    line *= ","
                end
                
                write(f, "$line\n")
            end
            
            write(f, ");\n\n")
        end
    end
    
    println("Script SQL gerado com sucesso em: $output_file")
end

# Check if file exists and run
input_json = "C:/Users/njunior/Documents/RHFP/MSPREV_SCHEMA_CREATE.json"
output_sql = "C:/Users/njunior/Documents/RHFP/MSPREV_DDL.sql"

if isfile(input_json)
    try
        generate_ddl(input_json, output_sql)
    catch e
        println("Erro durante o processamento: $e")
    end
else
    println("Erro: Arquivo '$input_json' não encontrado no diretório atual.")
end


# Check if file exists and run
input_json_2 = "C:/Users/njunior/Documents/RHFP/PREVI_SCHEMA_CREATE.json"
output_sql_2 = "C:/Users/njunior/Documents/RHFP/PREVI_DDL.sql"

if isfile(input_json_2)
    try
        generate_ddl(input_json_2, output_sql_2)
    catch e
        println("Erro durante o processamento: $e")
    end
else
    println("Erro: Arquivo '$input_json' não encontrado no diretório atual.")
end



# julia arquivosplit/src/gerar_ddl.jl