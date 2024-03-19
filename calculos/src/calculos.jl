module calculos
    using Printf
    greet() = print("Hello World!");

    RED         = "\033[1;31m";
    BLUE        = "\033[1;34m";
    CYAN        = "\033[1;36m";
    GREEN       = "\033[0;32m";
    VERDINHO    = "\032[1;32m";
    RESET       = "\033[0;0m";
    BOLD        = "\033[;1m";
    REVERSE     = "\033[;7m";

    include("fisica/formula.jl");
    function calcular()
        @printf("\n%s", "\e[4;31molá\e[m");

        m = parse(Float64, "100");
        a = parse(Float64, "9.81");
        n = forcaN(m, a);

        println(@sprintf("\e[1;36m\nA força aplicada é de \033[1;33m%.2f N\e[m para uma massa de \033[1;34m%.2f kg\e[m e uma aceleração de \033[1;35m%.2f m/s\e[m", n, m, a));

        formatted_forca = @sprintf("%.2f", n);
        formatted_massa = @sprintf("%.2f", m);
        formatted_aceleracao = @sprintf("%.2f", a);

        println("\e[1;44;33mA força\e[m \e[1;45;34maplicada é de\e[m \e[1;31m$(formatted_forca)N\e[m \e[1;43;32mpara uma massa de\e[m \e[1;31m$(formatted_massa)kg\e[m e uma aceleração de \e[0;31m$(formatted_aceleracao)m/s²\e[m");

        Peso = pesoN("100.0","Marte");
        # (peso,massa,get_aceleracao_gravitacional(espaco));
        gravidade = Peso[3][1];
        lugar = Peso[3][2];
        println(@sprintf("\e[1;31mO peso é\e[m \033[1;32m%.2fN\e[m , para a massa de \033[1;32m%.2f kg\e[m, a aceleração da gravidade é de  \033[1;32m%.8f m/s² %s\e[m", Peso[1], Peso[2], Peso[3][1], Peso[3][2]));

        println("\e[1;30m$gravidade $lugar  \n\e[m");
        println("\033[42m \033[1m \033[33m Isto eh amarelo negrito com fundo verde \033[0;0m");
        println("\033[1;32mVERDINHO  \e[m \n");
        exit();
    end
end # module calculos

# calculos.calcular();

# executar

# cd calculos/src/

# julia calculos.jl
#/julia_app/calculos/src$ julia
#               _
#   _       _ _(_)_     |  Documentation: https://docs.julialang.org
#  (_)     | (_) (_)    |
#   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
#  | | | | | | |/ _` |  |
#  | | |_| | | | (_| |  |  Version 1.9.3 (2023-08-24)
# _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
#|__/                   |

# julia>  include("calculos.jl"); calculos.calcular();


############ julia> import calculos  import Pkg; Pkg.add("calculos"); calculos.calcular();
############ julia> import Pkg; Pkg.add("calculos"); calculos.calcular();
