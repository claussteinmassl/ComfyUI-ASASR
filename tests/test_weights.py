"""Weight path resolution and auto-download behavior (no network)."""
from pathlib import Path
from unittest import mock

import pytest

from asasr import weights


def test_existing_files_skip_download(tmp_path):
    sr = tmp_path / "ASASR" / weights.SR_LORA_FILE
    dpo = tmp_path / "ASASR" / weights.DPO_LORA_FILE
    sr.parent.mkdir(parents=True)
    (tmp_path / "ASASR" / "dpo_lora").mkdir(parents=True, exist_ok=True)
    sr.write_bytes(b"x")
    dpo.parent.mkdir(parents=True, exist_ok=True)
    dpo.write_bytes(b"x")

    with mock.patch.object(weights, "hf_hub_download") as dl:
        paths = weights.ensure_lora_weights(base_dir=tmp_path, auto_download=True)

    dl.assert_not_called()
    assert paths.sr == sr
    assert paths.dpo == dpo


def test_missing_files_trigger_download(tmp_path):
    def fake_download(repo_id, filename, local_dir):
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
        return str(target)

    with mock.patch.object(weights, "hf_hub_download", side_effect=fake_download) as dl:
        paths = weights.ensure_lora_weights(base_dir=tmp_path, auto_download=True)

    assert dl.call_count == 2
    called = {c.kwargs["filename"] for c in dl.call_args_list}
    assert called == {weights.SR_LORA_FILE, weights.DPO_LORA_FILE}
    for c in dl.call_args_list:
        assert c.kwargs["repo_id"] == "wafer-bob/ASASR"
    assert paths.sr.exists() and paths.dpo.exists()


def test_missing_files_without_autodownload_raise(tmp_path):
    with pytest.raises(FileNotFoundError, match="sr_lora"):
        weights.ensure_lora_weights(base_dir=tmp_path, auto_download=False)
