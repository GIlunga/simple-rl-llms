MODEL_NAME = "Qwen/Qwen3.5-0.8B"


def download_models() -> None:
    # Helper for Modal image caching
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_NAME)
