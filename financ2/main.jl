using Printf

#import Pkg; Pkg.add("WeakRefStrings");
#import Pkg; Pkg.add("SQLite_jll");
#import Pkg; Pkg.add("DBInterface");


#include("persist/SQLite_jl/src/SQLite.jl");

include("persist/conedatabase.jl");
include("persist/migration.jl");
include("persist/mvc.jl");
import Pkg;
Pkg.add("SQLite3")
Pkg.add("DataFrames")
import Pkg; Pkg.add("SQLite3")
import SQLite3

# Cria um banco de dados chamado "meu_banco.db"
db = SQLite3.open("meu_banco.db");

# Cria uma tabela chamada "pessoas"
db.create_table("pessoas",
               ["nome", String],
               ["idade", Integer],
               ["sexo", String]);

# Insere um registro na tabela "pessoas"
db.insert("pessoas", ["João", 20, "Masculino"]);

# Insere outro registro na tabela "pessoas"
db.insert("pessoas", ["Maria", 30, "Feminino"]);

# Lê todos os registros da tabela "pessoas"
registros = db.query("SELECT * FROM pessoas");

# Imprime os registros
for registro in registros
  println(registro);
end

# Fecha o banco de dados
db.close();