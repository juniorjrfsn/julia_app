# projeto : cnncheckin
# file : cnncheckin/src/checkin/config.jl

module ConfigParam
    # Caminhos de diretórios
    const DADOS_DIR = abspath("../../../../dados")
    const TRAIN_DATA_PATH = "$DADOS_DIR/fotos_train"
    const INCREMENTAL_DATA_PATH = "$DADOS_DIR/fotos_new"
    const AUTH_DATA_PATH = "$DADOS_DIR/fotos_auth"

    # Global Configuration
    const CONFIG = Dict(
        # Model parameters
        :img_size => (128, 128),
        :batch_size => 8,
        :epochs => 30,
        :learning_rate => 0.0001,
        
        # Paths (using absolute paths to avoid issues)
        :data_dir => abspath("$DADOS_DIR/webcamcnn"),
        :photos_dir => nothing,  # Will be set in init
        :models_dir => nothing,  # Will be set in init
        :visualizations_dir => nothing, # Will be set in init
        
        # Files
        :model_file => "face_model.jld2",
        :config_file => "system_config.toml",
        :weights_file => "model_weights.toml"
    )
end # module ConfigParam