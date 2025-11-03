# projeto: cnncheckin
# file: cnncheckin/src/checkin/pretrain.jl

# projeto: cnncheckin
# file: cnncheckin/src/cnncheckin_pretrain.jl
# descrição: Script para pré-treinamento do modelo CNN

module CheckinPretrain
    using Flux
    using Statistics
    using Random
    using JLD2
    using Dates
    using Logging
    include("config_lib.jl") # inclui o módulo Config
    include("cnncheckin_core.jl")
    using .CNNCheckinCore

    # ============================================================================
    # CARREGAMENTO DE DADOS
    # ============================================================================

    """
        load_training_data(data_path::String; use_augmentation::Bool=true) 
            -> Tuple{Vector{CNNCheckinCore.PersonData}, Vector{String}}

    Carrega os dados de treinamento inicial do diretório especificado.

    # Argumentos
    - `data_path::String`: Caminho para o diretório com imagens
    - `use_augmentation::Bool`: Se deve aplicar data augmentation

    # Retorna
    - `Vector{CNNCheckinCore.PersonData}`: Dados de cada pessoa
    - `Vector{String}`: Lista de nomes das pessoas
    """
    function load_training_data(data_path::String; use_augmentation::Bool=true)
        @info "📂 Carregando dados de treinamento..." path=data_path
        
        if !isdir(data_path)
            println(
                """
                    \n📂 Tentativa de conexão ao diretório de dados falhou! \n
                """
            )
            exit()
            throw(ArgumentError("Diretório não encontrado: $data_path"))
        end
        
        person_images = Dict{String, Vector{Array{Float32, 3}}}()
        processed_count = 0
        failed_count = 0
        
        # Processar todos os arquivos do diretório
        for filename in readdir(data_path)
            filepath = joinpath(data_path, filename)
            
            # Validar arquivo
            if !CNNCheckinCore.validate_image_file(filepath)
                failed_count += 1
                continue
            end
            
            try
                # Extrair nome da pessoa
                person_name = CNNCheckinCore.extract_person_name(filename)
                
                # Processar imagem
                img_arrays = CNNCheckinCore.preprocess_image(filepath; augment=use_augmentation)
                
                if img_arrays !== nothing && !isempty(img_arrays)
                    if !haskey(person_images, person_name)
                        person_images[person_name] = Array{Float32, 3}[]
                    end
                    
                    append!(person_images[person_name], img_arrays)
                    
                    aug_info = use_augmentation ? " ($(length(img_arrays)) variações)" : ""
                    @info "✅ Processado: $filename → $person_name$aug_info"
                    processed_count += 1
                else
                    @warn "❌ Falha ao processar: $filename"
                    failed_count += 1
                end
                
            catch e
                @error "Erro ao processar arquivo" filename=filename exception=(e, catch_backtrace())
                failed_count += 1
            end
        end
        
        # Resumo do carregamento
        @info """
        📊 Resumo do carregamento:
        - Arquivos processados: $processed_count
        - Arquivos falhados: $failed_count
        - Pessoas encontradas: $(length(person_images))
        """
        
        if isempty(person_images)
            throw(ArgumentError("Nenhuma imagem válida encontrada em: $data_path"))
        end
        
        # Criar estruturas PersonData
        people_data = CNNCheckinCore.PersonData[]
        person_names = sort(collect(keys(person_images)))
        
        for (idx, person_name) in enumerate(person_names)
            images = person_images[person_name]
            if !isempty(images)
                push!(people_data, CNNCheckinCore.PersonData(person_name, images, idx, false))
                @info "👤 $person_name: $(length(images)) imagens (Label: $idx)"
            end
        end
        
        return people_data, person_names
    end

    # ============================================================================
    # CRIAÇÃO DE DATASETS
    # ============================================================================

    """
        split_train_validation(people_data::Vector{CNNCheckinCore.PersonData}, split_ratio::Float64=0.8)
            -> Tuple{Tuple, Tuple}

    Divide os dados em conjuntos de treino e validação.

    # Retorna
    - Tupla (train_images, train_labels)
    - Tupla (val_images, val_labels)
    """
    function split_train_validation(people_data::Vector{CNNCheckinCore.PersonData}, split_ratio::Float64=0.8)
        @info "📊 Dividindo dados em treino e validação..." ratio=split_ratio
        
        train_images = Array{Float32, 3}[]
        train_labels = Int[]
        val_images = Array{Float32, 3}[]
        val_labels = Int[]
        
        for person in people_data
            n_imgs = length(person.images)
            n_train = max(1, Int(floor(n_imgs * split_ratio)))
            
            # Embaralhar índices
            indices = randperm(n_imgs)
            
            # Separar treino
            for i in 1:n_train
                push!(train_images, person.images[indices[i]])
                push!(train_labels, person.label)
            end
            
            # Separar validação
            for i in (n_train+1):n_imgs
                push!(val_images, person.images[indices[i]])
                push!(val_labels, person.label)
            end
            
            @info "  $(person.name): $n_train treino | $(n_imgs - n_train) validação"
        end
        
        @info """
        ✅ Datasets criados:
        - Treino: $(length(train_images)) imagens
        - Validação: $(length(val_images)) imagens
        """
        
        return (train_images, train_labels), (val_images, val_labels)
    end

    """
        create_batches(images, labels, batch_size::Int) -> Vector

    Cria batches para treinamento.
    """
    function create_batches(images, labels, batch_size::Int)
        batches = []
        n_samples = length(images)
        
        if n_samples == 0
            return batches
        end
        
        # Determinar range de labels
        unique_labels = unique(labels)
        label_range = minimum(unique_labels):maximum(unique_labels)
        
        @info "🏷️  Criando batches:" n_samples=n_samples batch_size=batch_size label_range=label_range
        
        # Criar batches
        for i in 1:batch_size:n_samples
            end_idx = min(i + batch_size - 1, n_samples)
            batch_images = images[i:end_idx]
            batch_labels = labels[i:end_idx]
            
            try
                # Concatenar imagens
                batch_tensor = cat(batch_images..., dims=4)
                
                # One-hot encoding dos labels
                batch_labels_onehot = Flux.onehotbatch(batch_labels, label_range)
                
                push!(batches, (batch_tensor, batch_labels_onehot))
                
            catch e
                @error "Erro ao criar batch" range="$i:$end_idx" exception=(e, catch_backtrace())
                continue
            end
        end
        
        @info "✅ Criados $(length(batches)) batches"
        return batches
    end

    # ============================================================================
    # ARQUITETURA DO MODELO
    # ============================================================================

    """
        build_cnn_model(num_classes::Int, input_size::Tuple{Int,Int}=IMG_SIZE) -> Chain

    Constrói a arquitetura CNN para reconhecimento facial.

    # Arquitetura
    - 4 blocos convolucionais com BatchNorm, Dropout e MaxPool
    - 3 camadas densas com Dropout
    - Camada de saída com num_classes neurônios
    """
    function build_cnn_model(num_classes::Int, input_size::Tuple{Int,Int}=CNNCheckinCore.IMG_SIZE)
        @info "🗺️  Construindo modelo CNN..." num_classes=num_classes input_size=input_size
        
        # Calcular dimensões após as camadas convolucionais
        final_h = input_size[1]
        final_w = input_size[2]
        
        for _ in 1:4  # 4 MaxPool layers
            final_h = div(final_h, 2)
            final_w = div(final_w, 2)
        end
        
        final_features = 256 * final_h * final_w
        
        @info "  Dimensões finais das features: $final_h × $final_w × 256 = $final_features"
        
        model = Chain(
            # Bloco 1: Extração de features básicas
            Conv((3, 3), 3 => 64, relu, pad=1),
            BatchNorm(64),
            Dropout(0.1),
            MaxPool((2, 2)),
            
            # Bloco 2: Features intermediárias
            Conv((3, 3), 64 => 128, relu, pad=1),
            BatchNorm(128),
            Dropout(0.1),
            MaxPool((2, 2)),
            
            # Bloco 3: Features avançadas
            Conv((3, 3), 128 => 256, relu, pad=1),
            BatchNorm(256),
            Dropout(0.15),
            MaxPool((2, 2)),
            
            # Bloco 4: Features de alto nível
            Conv((3, 3), 256 => 256, relu, pad=1),
            BatchNorm(256),
            Dropout(0.15),
            MaxPool((2, 2)),
            
            # Classificador
            Flux.flatten,
            Dense(final_features, 512, relu),
            Dropout(0.4),
            Dense(512, 256, relu),
            Dropout(0.3),
            Dense(256, num_classes)
        )
        
        @info "✅ Modelo construído com $(length(model)) camadas"
        return model
    end

    # ============================================================================
    # TREINAMENTO
    # ============================================================================

    """
        calculate_accuracy(model, data_loader) -> Float64

    Calcula a acurácia do modelo em um conjunto de dados.
    """
    function calculate_accuracy(model, data_loader)
        correct = 0
        total = 0
        
        for (x, y) in data_loader
            try
                ŷ = softmax(model(x))
                pred = Flux.onecold(ŷ)
                true_labels = Flux.onecold(y)
                correct += sum(pred .== true_labels)
                total += length(true_labels)
            catch e
                @debug "Erro ao calcular acurácia em batch" exception=e
                continue
            end
        end
        
        return total > 0 ? correct / total : 0.0
    end

    """
        train_model!(model, train_data, val_data, epochs::Int, learning_rate::Float64)
            -> Tuple{Vector{Float64}, Vector{Float64}, Float64, Int}

    Treina o modelo CNN.

    # Retorna
    - Vector de losses de treino
    - Vector de acurácias de validação
    - Melhor acurácia de validação
    - Epoch com melhor acurácia
    """
    function train_model!(model, train_data, val_data, epochs::Int, learning_rate::Float64)
        @info "🚀 Iniciando treinamento..." epochs=epochs learning_rate=learning_rate
        
        # Configurar otimizador
        optimizer = ADAM(learning_rate, (0.9, 0.999), 1e-8)
        opt_state = Flux.setup(optimizer, model)
        
        # Métricas
        train_losses = Float64[]
        val_accuracies = Float64[]
        best_val_acc = 0.0
        best_epoch = 0
        
        # Early stopping
        patience_counter = 0
        patience_limit = 10
        
        @info "📊 Iniciando $(epochs) epochs de treinamento..."
        
        for epoch in 1:epochs
            epoch_loss = 0.0
            num_batches = 0
            
            # Fase de treinamento
            for (x, y) in train_data
                try
                    loss, grads = Flux.withgradient(model) do m
                        ŷ = m(x)
                        Flux.logitcrossentropy(ŷ, y)
                    end
                    
                    Flux.update!(opt_state, model, grads[1])
                    epoch_loss += loss
                    num_batches += 1
                    
                catch e
                    @error "Erro no batch de treino" epoch=epoch exception=(e, catch_backtrace())
                    continue
                end
            end
            
            # Calcular métricas
            avg_loss = num_batches > 0 ? epoch_loss / num_batches : 0.0
            val_acc = calculate_accuracy(model, val_data)
            
            push!(train_losses, avg_loss)
            push!(val_accuracies, val_acc)
            
            # Atualizar melhor modelo
            if val_acc > best_val_acc
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0
            else
                patience_counter += 1
            end
            
            # Log de progresso
            if epoch % 3 == 0 || epoch == 1
                @info "📈 Epoch $epoch/$epochs" loss=round(avg_loss, digits=4) val_acc=round(val_acc*100, digits=2) best_acc=round(best_val_acc*100, digits=2) best_epoch=best_epoch
            end
            
            # Early stopping
            if patience_counter >= patience_limit
                @info "⏹️  Early stopping: sem melhoria por $patience_limit epochs"
                break
            end
            
            # Parar se atingir boa acurácia
            if val_acc >= 0.90
                @info "🎯 Excelente acurácia alcançada!"
                break
            end
        end
        
        @info """
        ✅ Treinamento concluído!
        - Melhor acurácia: $(round(best_val_acc*100, digits=2))%
        - Melhor epoch: $best_epoch
        - Epochs treinadas: $(length(val_accuracies))
        """
        
        return train_losses, val_accuracies, best_val_acc, best_epoch
    end

    # ============================================================================
    # SALVAMENTO DO MODELO
    # ============================================================================

    """
        save_model(model, person_names::Vector{String}, training_info::Dict) -> Bool

    Salva o modelo treinado e suas configurações.
    """
    function save_model(model, person_names::Vector{String}, training_info::Dict)
        @info "💾 Salvando modelo e configurações..."
        
        # Salvar modelo
        model_data = Dict(
            "model_state" => model,
            "person_names" => person_names,
            "model_type" => "pretrained",
            "timestamp" => string(Dates.now()),
            "training_info" => training_info
        )
        
        try
            jldsave(CNNCheckinCore.MODEL_PATH; model_data=model_data)
            @info "✅ Modelo salvo: $(CNNCheckinCore.MODEL_PATH)"
        catch e
            @error "Erro ao salvar modelo" exception=(e, catch_backtrace())
            return false
        end
        
        # Salvar metadados TOML
        metadata_saved = CNNCheckinCore.save_model_data_toml(
            model, 
            person_names, 
            CNNCheckinCore.MODEL_DATA_TOML_PATH
        )
        
        # Salvar configuração
        config = CNNCheckinCore.create_default_config()
        config["model"]["num_classes"] = length(person_names)
        config["model"]["augmentation_used"] = training_info["augmentation_used"]
        config["training"]["epochs_trained"] = training_info["epochs_trained"]
        config["training"]["final_accuracy"] = training_info["final_accuracy"]
        config["training"]["best_epoch"] = training_info["best_epoch"]
        config["data"]["person_names"] = person_names
        config["data"]["timestamp"] = string(Dates.now())
        
        config_saved = CNNCheckinCore.save_config(config, CNNCheckinCore.CONFIG_PATH)
        
        if metadata_saved && config_saved
            @info """
            ✅ Todos os arquivos salvos com sucesso!
            - Modelo: $(CNNCheckinCore.MODEL_PATH)
            - Configuração: $(CNNCheckinCore.CONFIG_PATH)
            - Metadados: $(CNNCheckinCore.MODEL_DATA_TOML_PATH)
            """
            return true
        else
            @warn "Modelo salvo, mas houve problemas com arquivos auxiliares"
            return false
        end
    end

    # ============================================================================
    # FUNÇÃO PRINCIPAL
    # ============================================================================

    """
        pretrain_command() -> Bool

    Executa o pipeline completo de pré-treinamento.
    """
    function pretrain_command()
        println("\n" * "="^70)
        println("🧠 SISTEMA DE RECONHECIMENTO FACIAL - PRÉ-TREINAMENTO")
        println("="^70 * "\n")
        
        start_time = time()
        
        try
            # 1. Carregar dados
            people_data, person_names = load_training_data(
                ConfigParam.TRAIN_DATA_PATH; 
                use_augmentation=true
            )
            
            num_classes = length(person_names)
            total_images = sum(length(p.images) for p in people_data)
            
            @info """
            📊 Dados carregados:
            - Pessoas: $num_classes
            - Imagens totais: $total_images
            - Pessoas: $(join(person_names, ", "))
            """
            
            # 2. Dividir em treino/validação
            (train_images, train_labels), (val_images, val_labels) = split_train_validation(people_data)
            
            # 3. Criar batches
            train_batches = create_batches(train_images, train_labels, CNNCheckinCore.BATCH_SIZE)
            val_batches = create_batches(val_images, val_labels, CNNCheckinCore.BATCH_SIZE)
            
            if isempty(train_batches)
                throw(ArgumentError("Não foi possível criar batches de treino!"))
            end
            
            # 4. Construir modelo
            model = build_cnn_model(num_classes)
            
            # 5. Treinar modelo
            train_losses, val_accuracies, best_val_acc, best_epoch = train_model!(
                model,
                train_batches,
                val_batches,
                CNNCheckinCore.PRETRAIN_EPOCHS,
                CNNCheckinCore.LEARNING_RATE
            )
            
            # 6. Calcular estatísticas
            end_time = time()
            duration_minutes = (end_time - start_time) / 60
            
            training_info = Dict(
                "epochs_trained" => length(val_accuracies),
                "final_accuracy" => best_val_acc,
                "best_epoch" => best_epoch,
                "total_training_images" => length(train_images),
                "total_validation_images" => length(val_images),
                "augmentation_used" => true,
                "duration_minutes" => duration_minutes
            )
            
            # 7. Salvar modelo
            success = save_model(model, person_names, training_info)
            
            # 8. Exibir resumo
            println("\n" * "="^70)
            println("🎉 PRÉ-TREINAMENTO CONCLUÍDO!")
            println("="^70)
            println("📊 Resultados Finais:")
            println("   ✓ Acurácia: $(round(best_val_acc*100, digits=2))%")
            println("   ✓ Melhor epoch: $best_epoch")
            println("   ✓ Epochs treinadas: $(training_info["epochs_trained"])")
            println("   ✓ Duração: $(round(duration_minutes, digits=1)) minutos")
            println("   ✓ Pessoas reconhecidas: $num_classes")
            println("\n📁 Arquivos gerados:")
            println("   • $(CNNCheckinCore.MODEL_PATH)")
            println("   • $(CNNCheckinCore.CONFIG_PATH)")
            println("   • $(CNNCheckinCore.MODEL_DATA_TOML_PATH)")
            println("\n🚀 Próximos passos:")
            println("   1. Aprendizado incremental: julia cnncheckin_incremental.jl")
            println("   2. Identificação: julia cnncheckin_identify.jl <imagem>")
            println("="^70 * "\n")
            
            # retorna ao Menu Principal
            return true
            
        catch e
            @error "Erro durante pré-treinamento" exception=(e, catch_backtrace())
            println("\n❌ Pré-treinamento falhou!")
            return false
        end
    end
end
# ============================================================================
# EXECUÇÃO
# ============================================================================

"""
if abspath(PROGRAM_FILE) == @__FILE__
    success = CheckinPretrain.pretrain_command()
    exit(success ? 0 : 1)
end
"""
# julia  pretrain.jl