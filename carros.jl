struct Carro
	nome::String
	modelo::String
	ano::Int64
end

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