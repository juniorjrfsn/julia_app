module financ 

    include("persist/conedatabase.jl");
    include("persist/migration.jl");
    include("persist/mvc.jl");

    #Pkg.add("SQLite")
    #Pkg.add("DataFrames")

    greet() = print("Hello Worldesssj!")

end # module financ
