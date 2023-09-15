
function forcaN(Massa,Aceleracao)
    (Massa*Aceleracao)
end

function get_aceleracao_gravitacional(espaco)
    if espaco == "Sol"
        return (274.13, "no Sol")
    elseif espaco == "Terra"
        return (9.819649737724951, "na Terra")
    elseif espaco == "Lua"
        return (1.625, "na Lua")
    elseif espaco == "Marte"
        return (3.72076, "em Marte")
    else
        return (0,"...")  # Ação padrão se o espaço não for reconhecido
    end
end


function pesoN(massa, espaco)
    # Converte as massas para números de ponto flutuante
    massa = parse(Float64, massa);
    aceleracao_gravitacional = Dict(
        "Sol" => 274.13,
        "Terra" => 9.819649737724951,
        "Lua" => 1.625,
        "Marte" => 3.72076,
    );
    # Calcula o peso
    peso = massa * get(aceleracao_gravitacional, espaco, "Ação padrão");
    # Retorna o peso
    return (peso,massa,get_aceleracao_gravitacional(espaco));
end