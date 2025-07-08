import os
from huggingface_hub import hf_hub_download


def load_custom_model_from_hf(
    repo_id: str,
    model_filename: str = "pytorch_model.bin",
    config_filename: str = None,
    cache_dir: str = "./checkpoints"
) -> tuple:
    os.makedirs(cache_dir, exist_ok=True)
    model_path = hf_hub_download(repo_id=repo_id, filename=model_filename, cache_dir=cache_dir)
    if config_filename is None:
        return model_path
    config_path = hf_hub_download(repo_id=repo_id, filename=config_filename, cache_dir=cache_dir)

    return model_path, config_path