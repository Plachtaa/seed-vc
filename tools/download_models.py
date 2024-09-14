import os

from huggingface_hub import hf_hub_download


def check_and_download_files(
    repo_id: str,
    local_dir: str,
    file_list: list[str],
) -> None:
    os.makedirs(local_dir, exist_ok=True)
    for file in file_list:
        file_path = os.path.join(local_dir, file)
        if not os.path.exists(file_path):
            hf_hub_download(
                repo_id=repo_id,
                filename=file,
                local_dir=local_dir,
            )


repo_id = "Plachta/Seed-VC"
local_dir = "./checkpoints"
files = [
    "DiT_step_315000_seed_v2_online_pruned.pth",
    "README.md",
    "config_dit_mel_seed_wavenet.yml",
    "hifigan.yml",
    "hift.pt",
    "speech_tokenizer_v1.onnx",
]

check_and_download_files(repo_id, local_dir, files)
