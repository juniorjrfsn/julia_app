
module calculos
    using Printf
    greet() = print("Hello World!")

    include("fisica/formula.jl");
    function calcular()
        @printf("\n%s", "olá");

        m = parse(Float64, "100");
        a = parse(Float64, "9.81");
        n = forcaN(m, a);

    println(@sprintf("\e[1;32m\nA força aplicada é de %.2f N para uma massa de %.2f kg e uma aceleração de %.2f m/s\e[m", n, m, a))

        formatted_forca = @sprintf("%.2f", n);
        formatted_massa = @sprintf("%.2f", m);
        formatted_aceleracao = @sprintf("%.2f", a);

    println("\e[1;44;33mA força aplicada é de $(formatted_forca)N para uma massa de $(formatted_massa)kg e uma aceleração de $(formatted_aceleracao)m/s²\e[m")

        Peso = pesoN("100.0","Marte");
        # (peso,massa,get_aceleracao_gravitacional(espaco));
        gravidade = Peso[3][1];
        lugar = Peso[3][2];
    println(@sprintf("\e[1;31mO peso é %.2fN, para a massa de %.2f kg, a aceleração da gravidade é de  %.8f m/s² %s \e[m", Peso[1], Peso[2], Peso[3][1], Peso[3][2]))

        println("$gravidade $lugar  \n");
        exit()
    end
end # module calculos

# calculos.calcular();

# executar
# julia calculos.jl
