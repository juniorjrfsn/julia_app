# projeto : cnncheckin
# file : cnncheckin/src/checkin/main.jl

module Checkin
    
    println("\n⚙️   Carregando módulo Checkin...\n") 
    include("config.jl") # inclui o módulo Config
    include("cnncheckin_core.jl")
    using .CNNCheckinCore
    include("menu.jl") # inclui o módulo Menu 
    include("pretrain.jl") # inclui o módulo CheckinPretrain
    include("incremental.jl") # inclui o módulo CheckinIncremental.
    include("identif.jl") # inclui o módulo Identif.
    println("\n⚙️   Módulo Checkin carregado com sucesso.")

    function main()
        args = Base.ARGS  # Importa ARGS do Base
        if length(args) == 0 
            Menu.run_menu([
                "🚀 Iniciar Pré-treino",
                "📸 Iniciar captura de imagens e treino incremental",
                "🖼️  Iniciar treino incremental sem captura (usar imagens existentes)",
                "💽 Iniciar sistema de identificação",
                "📲 Sair"
            ]; handlers=Dict(
                1 => () -> begin
                    println("🚀 Iniciando Pré treino") 
                    success = CheckinPretrain.pretrain_command()
                    if success
                        println("✅ Pré-treinamento concluído com sucesso!")
                    else
                        println("❌ Falha no pré-treinamento.")
                    end
                    print("\nPressione ENTER para continuar...")
                    readline()
                    #exit(success ? 0 : 1)
                end,
                2 => () -> begin
                    println("📸 Iniciando captura de imagens e treino incremental")
                    println("\n" * "="^70)
                    println("📸 CAPTURA E TREINO INCREMENTAL")
                    println("="^70)
                    println("⚠️  Funcionalidade em desenvolvimento")
                    println("\nRecursos planejados:")
                    println("  • Captura de imagens via webcam")
                    println("  • Treino incremental automático")
                    println("  • Interface de captura interativa")
                    print("\nPressione ENTER para continuar...")
                    readline()
                   # incremental_learning_with_capture_command()
                end,
                3 => () -> begin
                    println("🖼️ Iniciando treino incremental sem captura")
                    success = Increment.incremental_learning_command()
                    if success
                        println("\n✅ Treino incremental concluído com sucesso!")
                    else
                        println("\n❌ Falha no treino incremental.")
                    end
                    print("\nPressione ENTER para continuar...")
                    readline()
                    # exit(success ? 0 : 1)
                    # incremental_learning_command()
                end,
                4 => () -> begin
                    println("\n" * "="^70)
                    println("💽 SISTEMA DE IDENTIFICAÇÃO")
                    println("Iniciando identificação (webcam)")
                    println("="^70)
                    Identif.show_identification_menu()
                  #  incremental_learning_command()
                end,
                5 => () -> begin
                    println("\n" * "="^70)
                    println("👋 Saindo do cnncheckin. Até mais!")
                    println("="^70 * "\n")
                end
            ), loop=true)
             elseif args[1] == "--pretrain" || args[1] == "-p"
            # Pré-treino via linha de comando
            println("🚀 Iniciando pré-treinamento via linha de comando")
            success = CheckinPretrain.pretrain_command()
            exit(success ? 0 : 1)
            
        elseif args[1] == "--incremental" || args[1] == "-i"
            # Treino incremental via linha de comando
            println("🖼️  Iniciando treino incremental via linha de comando")
            success = Increment.incremental_learning_command()
            exit(success ? 0 : 1)
            
        elseif args[1] == "--identify" || args[1] == "-d"
            # Identificação via linha de comando
            if length(ARGS) < 2
                println("❌ Erro: especifique o caminho da imagem")
                println("Uso: julia main.jl --identify <caminho_da_imagem>")
                exit(1)
            end
            
            println("💽 Identificando imagem via linha de comando")
            # Passar argumentos para o módulo Identif
            global ARGS = ARGS[2:end]
            Identif.main()
            
        elseif args[1] == "--help" || args[1] == "-h"
            println("""
            ╔════════════════════════════════════════════════════════════════╗
            ║                    CNNCHECKIN - AJUDA                          ║
            ╚════════════════════════════════════════════════════════════════╝
            
            DESCRIÇÃO:
              Sistema de reconhecimento facial com CNN usando Julia/Flux
            
            USO:
              julia main.jl [OPÇÃO] [ARGUMENTOS]
            
            OPÇÕES:
              (sem opção)           Inicia menu interativo
              -h, --help            Mostra esta ajuda
              -p, --pretrain        Executa pré-treinamento
              -i, --incremental     Executa treino incremental
              -d, --identify <img>  Identifica pessoa em imagem
              -q, --quit            Sai do programa
            
            EXEMPLOS:
              # Menu interativo
              julia main.jl
              
              # Pré-treinamento
              julia main.jl --pretrain
              
              # Treino incremental
              julia main.jl --incremental
              
              # Identificar imagem
              julia main.jl --identify foto.jpg
              
              # Identificar com autenticação
              julia main.jl --identify foto.jpg --auth "João Silva"
              
              # Identificação em lote
              julia main.jl --identify --batch ./fotos/
            
            ESTRUTURA DE DIRETÓRIOS:
              dados/
              ├── fotos_train/      # Imagens para pré-treinamento
              │   ├── joao-1.jpg
              │   ├── joao-2.jpg
              │   ├── maria-1.jpg
              │   └── ...
              ├── fotos_new/        # Imagens para treino incremental
              │   └── ...
              └── fotos_auth/       # Imagens para teste/identificação
                  └── ...
            
            FORMATO DE ARQUIVOS:
              • Nome: nome-numero.extensao (ex: joao-1.jpg)
              • Formatos: .jpg, .jpeg, .png, .bmp, .tiff, .gif, .webp
              • Tamanho: entre 500 bytes e 50 MB
              • Dimensões: mínimo 10x10 pixels
            
            WORKFLOW RECOMENDADO:
              1. Organize fotos em dados/fotos_train/
              2. Execute: julia main.jl --pretrain
              3. Adicione novas pessoas em dados/fotos_new/
              4. Execute: julia main.jl --incremental
              5. Teste: julia main.jl --identify teste.jpg
            
            MAIS INFORMAÇÕES:
              • Documentação: README.md
              • Repositório: [seu repositório]
              • Issues: [seu issue tracker]
            
            ════════════════════════════════════════════════════════════════
            """)
        elseif args[1] == "--capture"
            # Treinar com captura
            println("🚀 Modo captura e treino incremental")
        elseif args[1] == "--quit" || args[1] == "-q"
            # Sair do programa
            println("👋 Saindo do cnncheckin. Até mais!")
        elseif args[1] == "--no-capture"
            # Treinar sem captura (usar imagens existentes)
            println("📚 Modo incremental sem captura")
            incremental_learning_command()
            
 
        else
            println("❌ Opção desconhecida: $(args[1])")
            println("Use --help para ver as opções disponíveis")
            exit(1)
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