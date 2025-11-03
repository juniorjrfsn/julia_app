# projeto : cnncheckin
# file : cnncheckin/src/checkin/menu.jl
module Menu

    export run_menu, show_menu, prompt_choice

    """
        show_menu(options::Vector{String})

    Mostra o menu de opções (lista numerada).
    """
    function show_menu(options::Vector{String})
        println("\n=== Menu de Opções ===")
        for (i, opt) in enumerate(options)
            println("$(i).  $opt")
        end
    end

    """
        prompt_choice(n::Integer) -> Int

    Pede ao utilizador uma escolha válida entre 1 e n.
    Bloqueia até receber uma escolha válida.
    """
    function prompt_choice(n::Integer)
        while true
            print("Escolha uma opção (1-$n): ")
            flush(stdout)
            input = try
                readline()
            catch
                return n  # se EOF, retorna última opção (tipicamente "Sair")
            end
            s = strip(input)
            try
                c = parse(Int, s)
                if 1 <= c <= n
                    return c
                else
                    println("Opção inválida. Insira um número entre 1 e $n.")
                end
            catch
                println("Entrada inválida. Digite um número.")
            end
        end
    end

    """
        run_menu(options::Vector{String}; handlers=Dict{Int,Function}(), loop=true)

    Executa o menu. `options` é a lista de rótulos. `handlers` é um Dict que mapeia índices
    para funções a executar quando a opção é escolhida. Se não houver handler para uma opção,
    apenas imprime a opção selecionada. Se `loop` for true, o menu repete; por convenção a última
    opção pode ser usada para "Sair".
    """
    function run_menu(options::Vector{String}; handlers=Dict{Int,Function}(), loop::Bool=true)
        n = length(options)
        while true
            show_menu(options)
            choice = prompt_choice(n)
            if haskey(handlers, choice)
                try
                    handlers[choice]()
                catch err
                    println("Erro ao executar handler: ", err)
                end
            else
                println("Opção $(choice) selecionada: ", options[choice])
            end
            if !loop || choice == n
                break
            end
        end
    end

end # module