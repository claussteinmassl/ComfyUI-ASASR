"""Resolution and auto-download of the ASASR LoRA weights.

The two inference LoRAs live on Hugging Face at ``wafer-bob/ASASR``. They are
stored under ``<loras>/ASASR/`` using the upstream relative filenames so that
manually downloaded weights are picked up as well.
"""
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

HF_REPO_ID = "wafer-bob/ASASR"
SR_LORA_FILE = "sr_lora/pytorch_lora_weights_v2.safetensors"
DPO_LORA_FILE = "dpo_lora/adapter_model.safetensors"


@dataclass(frozen=True)
class LoraPaths:
    """Local filesystem paths of the two ASASR LoRA weight files."""

    sr: Path
    dpo: Path


def default_base_dir() -> Path:
    """Returns ``<ComfyUI loras dir>/ASASR`` (requires running inside ComfyUI).

    Raises:
        RuntimeError: If ComfyUI's ``folder_paths`` module is unavailable.
    """
    try:
        import folder_paths
    except ImportError as exc:  # pragma: no cover - exercised only in ComfyUI
        raise RuntimeError(
            "folder_paths is unavailable; pass base_dir explicitly."
        ) from exc
    return Path(folder_paths.get_folder_paths("loras")[0]) / "ASASR"


def ensure_lora_weights(base_dir: Path | None = None, auto_download: bool = True) -> LoraPaths:
    """Ensures both LoRA files exist locally, downloading them if allowed.

    Args:
        base_dir: Parent directory; weights are read from and written to
            ``<base_dir>/ASASR/...``. Defaults to the ComfyUI loras directory,
            giving ``<loras>/ASASR/...``.
        auto_download: Download missing files from Hugging Face when True.

    Returns:
        The resolved local paths.

    Raises:
        FileNotFoundError: A file is missing and ``auto_download`` is False.
    """
    # Invariant: base always ends in .../ASASR, whether from default_base_dir() or by appending to the caller's parent dir.
    base = Path(base_dir) / "ASASR" if base_dir is not None else default_base_dir()
    paths = LoraPaths(sr=base / SR_LORA_FILE, dpo=base / DPO_LORA_FILE)
    for path, filename in ((paths.sr, SR_LORA_FILE), (paths.dpo, DPO_LORA_FILE)):
        if path.exists():
            continue
        if not auto_download:
            raise FileNotFoundError(
                f"ASASR weight file missing: {path}. Download {filename} from "
                f"https://huggingface.co/{HF_REPO_ID} or enable auto_download."
            )
        print(f"[ASASR] downloading {filename} from {HF_REPO_ID} ...")
        hf_hub_download(repo_id=HF_REPO_ID, filename=filename, local_dir=str(base))
    return paths
