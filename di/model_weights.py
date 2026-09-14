"""Helpers for resolving model checkpoints (local → Hugging Face fallback).

Every non-base model variant in this repo (TabDPT seed/split variants,
NanoTabPFN classifier) ships its weights from the ``dwahdany/tfms`` HF
repo. ``ensure_local_weights`` returns a usable filesystem path,
downloading from HF on first use and caching under the default HF cache
afterwards. Once the file is on disk the function is a no-op.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

DEFAULT_HF_REPO = "dwahdany/tfms"


def ensure_local_weights(
    local_path: str | Path | None,
    hf_path: Optional[str] = None,
    hf_repo: str = DEFAULT_HF_REPO,
) -> str:
    """Return a path to the checkpoint, downloading from HF if missing.

    Parameters
    ----------
    local_path :
        Preferred on-disk location. If the file already exists, it is
        returned verbatim — no network call.
    hf_path :
        Path of the file inside ``hf_repo``. Required for the download
        fallback; if unset and ``local_path`` does not exist, this raises.
    hf_repo :
        Hugging Face repository id. Defaults to ``dwahdany/tfms``.
    """
    if local_path is not None:
        p = Path(local_path)
        if p.exists():
            return str(p)

    if hf_path is None:
        raise FileNotFoundError(
            f"Weights not found at {local_path!r} and no hf_path provided"
        )

    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=hf_repo, filename=hf_path)
