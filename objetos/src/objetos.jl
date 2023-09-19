module objetos

    greet() = print("Hello World!")
    struct Carro
        nome::String
        modelo::String
        ano::Int64
    end
    function geraCarros()
        carros = [
            Carro("Fiat", 		"Uno", 	2023),
            Carro("Volkswagen", "Gol", 	2022),
            Carro("Hyundai", 	"HB20", 2021)
        ]
        println("----------------------")
        println(1+1)
        println('1','1')
        println("----------------------")
        for carro in carros
            println(carro.nome, carro.modelo, carro.ano)
        end
        println("----------------------")
        println("Selecionando o segundo carro")
        segundo_carro = carros[2]
        println("Marca: $(segundo_carro.nome), Modelo: $(segundo_carro.modelo), Ano: $(segundo_carro.ano)")
        println("----------------------")
    end

end # module objetos
