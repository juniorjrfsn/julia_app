# projeto : cnncheckin
# file : cnncheckin/src/checkin/main.jl

module Checkin
    
    println("\n⚙️   Carregando módulo Checkin...\n") 
    include("config.jl") # inclui o módulo Config
    include("cnncheckin_core.jl")
    using .CNNCheckinCore
    include("menu.jl") # inclui o módulo Menu 
    include("pretrain.jl") # inclui o módulo CheckinPretrain
    include("incremental.jl") # inclui o módulo CheckinIncremental
    println("\n⚙️   Módulo Checkin carregado com sucesso.")

    function main()
        if length(ARGS) == 0 
            Menu.run_menu([
                "🚀-Iniciar Pré treino",
                "📸-Iniciar captura de imagens e treino incremental",
                "🖼️ -Iniciar treino incremental sem captura (usar imagens existentes)",
                "💽-Iniciar identificação (webcam)",
                "📲-Sair"
            ]; handlers=Dict(
                1 => () -> begin
                    println("🚀 Iniciando Pré treino") 
                    success = CheckinPretrain.pretrain_command()
                    if success
                        println("✅ Pré-treinamento concluído com sucesso!")
                    else
                        println("❌ Falha no pré-treinamento.")
                    end
                    #exit(success ? 0 : 1)
                end,
                2 => () -> begin
                    println("📸 Iniciando captura de imagens e treino incremental")
                   # incremental_learning_with_capture_command()
                end,
                3 => () -> begin
                    println("🖼️ Iniciando treino incremental sem captura")
                    success = Increment.incremental_learning_command()
                    if success
                        println("✅ Pré-treinamento concluído com sucesso!")
                    else
                        println("❌ Falha no pré-treinamento.")
                    end
                    #exit(success ? 0 : 1)
                  #  incremental_learning_command()
                end,
                4 => () -> begin
                    println("💽 Iniciando identificação (webcam)")
                  #  incremental_learning_command()
                end,
                5 => () -> begin
                    println("👋 Saindo do cnncheckin. Até mais!")
                end
            ), loop=true)
        elseif ARGS[1] == "--capture"
            # Treinar com captura
            println("🚀 Modo captura e treino incremental")
        elseif ARGS[1] == "--quit" || ARGS[1] == "-q"
            # Sair do programa
            println("👋 Saindo do cnncheckin. Até mais!")
        elseif ARGS[1] == "--no-capture"
            # Treinar sem captura (usar imagens existentes)
            println("📚 Modo incremental sem captura")
            incremental_learning_command()
            
        elseif ARGS[1] == "--help" || ARGS[1] == "-h"
            println("""
            ajuda do cnncheckin:
            """)
        else
            println("❌ Opção desconhecida: $(ARGS[1])")
            println("Use --help para ver as opções disponíveis")
        end
    end
end # module Checkin


# ============================================================================
# EXECUÇÃO
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    Checkin.main()
end

"""
  julia main.jl --help
  julia main.jl --no-capture
  julia main.jl --quit
  julia main.jl --unknown
"""