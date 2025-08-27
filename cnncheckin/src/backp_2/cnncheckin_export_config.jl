module cnncheckin_export_config

using Dates
import .cnncheckin_core

# Função para exportar configuração
function export_config_command(output_path::String = "modelo_config_export.toml")
    println("📤 Exportando configuração do modelo...")
    
    if !isfile(cnncheckin_core.CONFIG_PATH)
        println("❌ Configuração não encontrada. Execute primeiro: julia cnncheckin_train.jl")
        return false
    end
    
    try
        config = cnncheckin_core.load_config(cnncheckin_core.CONFIG_PATH)
        config["export"] = Dict(
            "exported_at" => string(Dates.now()),
            "exported_from" => cnncheckin_core.CONFIG_PATH,
            "export_version" => "1.0"
        )
        success = cnncheckin_core.save_config(config, output_path)
        if success
            println("✅ Configuração exportada para: $output_path")
            println("📋 Você pode editar este arquivo e importá-lo depois")
        end
        return success
    catch e
        println("❌ Erro ao exportar configuração: $e")
        return false
    end
end

# Executar comando
if abspath(PROGRAM_FILE) == @__FILE__
    output_path = length(ARGS) >= 1 ? ARGS[1] : "modelo_config_export.toml"
    success = export_config_command(output_path)
    if success
        println("✅ Exportação de configuração concluída!")
    else
        println("💥 Falha na exportação")
    end
end

end # module cnncheckin_export_config
