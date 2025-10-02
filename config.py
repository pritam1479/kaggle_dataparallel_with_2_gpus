from pathlib import Path

def get_config():
    return {
        # 'datasource':"Helsinki-NLP/opus_books",
        'datasource':"Helsinki-NLP/opus-100",
        'source_lang': 'en',
        'target_lang': 'it',
        'target_lang': 'hi',
        'tokenizer_file': 'tokenizer_{lang}.json',
        # 'max_seq_length': 350,
        'max_seq_length': 460,
        'num_layers': 2,
        'd_model': 512,
        'num_heads': 8,
        'd_ff': 512,
        'dropout_rate': 0.1,
        'batch_size': 16,
        'num_epochs': 20,
        'learning_rate': 1e-4,
        'device': 'cuda',  # or 'cpu'
        "model_folder": "weights",
        "model_basename": "transformer",
        "preload": "latest",
        "experiment_name": "runs/transformer_experiment"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])