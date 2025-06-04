from pathlib import Path

def get_config():
    return {
        "batch_size": 16,
        "num_epochs": 1,
        "lr": 5e-4,
        "seq_len": 128,
        "d_model": 256,
        "datasource": 'cfilt/iitb-english-hindi',
        "lang_src": "en",
        "lang_tgt": "hi",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,  # Make sure it starts clean
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/fast_test"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
