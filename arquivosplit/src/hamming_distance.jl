module HammingDistance
    function calcular(str1::String, str2::String)
        if length(str1) != length(str2)
            error("Strings devem ter o mesmo comprimento")
        end

        distance = 0
        for (char1, char2) in zip(str1, str2)
            distance += char1 != char2  # A conversão para inteiro é implícita
        end
        return distance
    end
end

# Para usar a função, você deve referenciar o módulo primeiro:
str1 = "Julia"
str2 = "Julis"
println(HammingDistance.calcular(str1, str2)) # Saída será 1 porque há uma diferença na posição do 'a' e 's'
