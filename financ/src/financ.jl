module financ

    include("persist/conedatabase.jl");
    include("persist/migration.jl");
    include("persist/mvc.jl");

    greet() = print("Bora galera!")

end # module financ
