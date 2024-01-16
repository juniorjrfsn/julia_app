module yinyang

    function calcular()

        for (j, k, l) in zip([1 2 3 4], [4 5 6 7], [4 5 6 7])
            println((j,k,l))
        end

        println("\033[1;32mVERDINHO  \e[m \n");
        exit();
    end
end



yinyang.calcular();

# executar
# julia yin_yang.jl