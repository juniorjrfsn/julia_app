# projeto: webcamcnn
# file: webcamcnn/src/integration_with_visualizations.jl

# Script para integrar visualizações de camadas ao sistema principal

println("🔄 Integrando sistema de visualizações de camadas...")

# Modificações necessárias no main_toml_system.jl
function add_visualization_menu_to_main()
    visualization_menu_code = """
        elseif choice == "7"
            println("\\n🎨 MODO: VISUALIZAÇÕES DE CAMADAS")
            println("=" ^ 40)
            visualization_management_menu()
    """
    
    println("📝 Para integrar as visualizações ao sistema principal:")
    println("1. Adicione esta linha no início do main_toml_system.jl:")
    println("   include(\"layer_visualization.jl\")")
    println("   include(\"pretrain_modified_with_visualization.jl\")")
    println()
    println("2. No main_menu(), adicione a opção 7:")
    println("   7️⃣ - Visualizações de camadas (análise detalhada)")
    println()
    println("3. No switch das opções, adicione:")
    println(visualization_menu_code)
end

# Função para verificar dependências adicionais
function check_visualization_dependencies()
    println("🔍 Verificando dependências para visualizações...")
    
    required_packages = [
        "ImageTransformations", 
        "Colors",
        "ColorTypes",
        "ImageFiltering"
    ]
    
    missing_packages = String[]
    
    for pkg in required_packages
        try
            eval(Meta.parse("using $pkg"))
            println("✅ $pkg")
        catch
            push!(missing_packages, pkg)
            println("❌ $pkg - Ausente")
        end
    end
    
    if !isempty(missing_packages)
        println("📦 Execute para instalar pacotes ausentes:")
        println("using Pkg; Pkg.add([\"$(join(missing_packages, "\", \""))\"])")
        return false
    end
    
    println("✅ Todas as dependências estão disponíveis!")
    return true
end

# Função para criar estrutura inicial de visualizações
function setup_visualization_system()
    println("🏗️ Configurando sistema de visualizações...")
    
    # Criar diretório base
    viz_path = "../../../dados/webcamcnn/layer_visualizations"
    if !isdir(viz_path)
        mkpath(viz_path)
        println("📁 Criado diretório base: $viz_path")
    end
    
    # Criar arquivo de configuração de visualizações
    config_content = """
# Configurações de Visualização de Camadas
# webcamcnn visualization config

[visualization]
enabled = true
save_frequency = 5  # Salvar a cada N épocas
max_filters_per_layer = 16
image_size = [64, 64]
output_format = "png"

[layers]
# Camadas a serem visualizadas (índices)
conv_layers = [1, 5, 9, 13]  # Posições aproximadas das camadas Conv
dense_layers = [17, 19, 21]  # Posições aproximadas das camadas Dense

[storage]
max_visualizations = 10  # Máximo de conjuntos de visualização a manter
cleanup_old = true
"""
    
    config_path = joinpath(viz_path, "viz_config.toml")
    if !isfile(config_path)
        open(config_path, "w") do f
            write(f, config_content)
        end
        println("⚙️ Criado arquivo de configuração: $config_path")
    end
    
    # Criar README explicativo
    readme_content = """
# Sistema de Visualização de Camadas CNN

Este diretório contém visualizações das camadas da rede neural durante o treinamento.

## Estrutura dos Arquivos

```
layer_visualizations/
├── viz_config.toml           # Configurações do sistema
├── training_YYYYMMDD_HHMMSS/ # Visualizações por treinamento
│   ├── pessoa1/
│   │   ├── layer_1/          # Filtros e ativações da camada 1
│   │   ├── layer_2/          # Filtros e ativações da camada 2
│   │   └── ...
│   ├── pessoa2/
│   └── README.md            # Índice do treinamento
└── training_report_*.md     # Relatórios de análise
```

## Tipos de Arquivos

### Ativações (`*_activation.png`)
- Mostram como a rede "vê" cada imagem após passar pela camada
- Úteis para entender o que a rede aprendeu

### Filtros (`*_filter*.png`)
- Mostram os padrões que cada filtro detecta
- Cada filtro é especializado em detectar características específicas

## Como Interpretar

### Camadas Iniciais (1-2)
- Detectam características básicas: bordas, linhas, texturas
- Filtros devem ser diversos e bem definidos

### Camadas Intermediárias (3-4)  
- Combinam características básicas em padrões mais complexos
- Começam a detectar partes de faces (olhos, nariz, boca)

### Camadas Finais (Dense)
- Criam representação abstrata para classificação
- Visualizadas como mapas de "importância"

## Indicadores de Problemas

❌ **Filtros Mortos**: Todos os pixels pretos (não aprendeu nada)
❌ **Filtros Duplicados**: Vários filtros muito similares (redundância)  
❌ **Ativações Uniformes**: Sem variação entre diferentes imagens
❌ **Ruído Excessivo**: Padrões caóticos sem estrutura clara

## Uso Recomendado

1. Compare visualizações entre épocas para ver evolução
2. Identifique problemas de aprendizado precocemente
3. Ajuste hiperparâmetros com base nas visualizações
4. Documente padrões interessantes para análise posterior
"""
    
    readme_path = joinpath(viz_path, "README.md")
    if !isfile(readme_path)
        open(readme_path, "w") do f
            write(f, readme_content)
        end
        println("📚 Criado README explicativo: $readme_path")
    end
    
    return true
end

# Função para teste do sistema de visualização
function test_visualization_system()
    println("🧪 Testando sistema de visualização...")
    
    try
        # Teste básico - criar um modelo simples
        test_model = Chain(
            Conv((3, 3), 3 => 8, relu, pad=1),
            MaxPool((2, 2)),
            Conv((3, 3), 8 => 16, relu, pad=1), 
            MaxPool((2, 2)),
            Flux.flatten,
            Dense(16*32*32, 10)  # Assumindo entrada 128x128
        )
        
        println("✅ Modelo de teste criado")
        
        # Teste de criação de diretórios
        test_persons = ["TestePessoa1", "TestePessoa2"]
        test_viz_path = joinpath("../../../dados/webcamcnn/layer_visualizations", "test")
        
        create_visualization_directories(test_viz_path, test_persons)
        println("✅ Criação de diretórios testada")
        
        # Limpeza do teste
        if isdir(test_viz_path)
            rm(test_viz_path, recursive=true)
            println("🧹 Limpeza do teste concluída")
        end
        
        return true
        
    catch e
        println("❌ Erro no teste: $e")
        return false
    end
end

# Script principal de integração
function main_integration()
    println("🚀 INTEGRAÇÃO DO SISTEMA DE VISUALIZAÇÕES")
    println("=" ^ 50)
    
    # Verificar dependências
    if !check_visualization_dependencies()
        println("❌ Dependências ausentes. Instale os pacotes necessários primeiro.")
        return false
    end
    
    # Configurar sistema
    if !setup_visualization_system()
        println("❌ Falha na configuração do sistema")
        return false
    end
    
    # Teste do sistema
    if !test_visualization_system()
        println("❌ Falha no teste do sistema")
        return false
    end
    
    # Instruções finais
    println("\n✅ SISTEMA DE VISUALIZAÇÕES INTEGRADO COM SUCESSO!")
    println("\n📋 Próximos passos:")
    println("1. Execute o sistema principal: julia main_toml_system.jl")
    println("2. Escolha a opção '7 - Visualizações de camadas'")
    println("3. Execute um treinamento com visualizações")
    println("\n💡 Dicas:")
    println("- Visualizações são salvas automaticamente durante o treino")
    println("- Use frequência 3-5 épocas para treinamentos longos")
    println("- Analise as visualizações após cada treinamento")
    
    # Mostrar instruções de integração
    add_visualization_menu_to_main()
    
    return true
end

# Executar integração se chamado diretamente
if abspath(PROGRAM_FILE) == @__FILE__
    success = main_integration()
    if success
        println("\n🎉 Integração concluída com sucesso!")
        println("Execute: julia main_toml_system.jl para usar o sistema completo")
    else
        println("\n❌ Falha na integração. Verifique os erros acima.")
    end
end

export main_integration, check_visualization_dependencies, setup_visualization_system