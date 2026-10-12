"""Runtime settings, resolved once from the environment."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

DEVICES = ("auto", "cpu", "cuda")
_TRUTHY = {"1", "true", "yes", "on"}


def _cuda_device_count() -> int:
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


def _default_output_dir(env: Mapping[str, str]) -> Path:
    data_home = env.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(data_home) / "quotable"


@dataclass(frozen=True)
class Settings:
    output_dir: Path
    model: str
    device: str
    compute_type: str
    keep_audio: bool = False
    hf_token: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        cuda_count: Callable[[], int] = _cuda_device_count,
    ) -> Settings:
        env = os.environ if env is None else env

        device = (env.get("QUOTABLE_DEVICE") or "auto").strip().lower()
        if device not in DEVICES:
            raise ValueError(
                f"QUOTABLE_DEVICE={device!r} is invalid; use one of {', '.join(DEVICES)}"
            )
        if device == "auto":
            device = "cuda" if cuda_count() > 0 else "cpu"

        on_cuda = device == "cuda"
        model = env.get("QUOTABLE_MODEL") or ("turbo" if on_cuda else "base")
        compute_type = env.get("QUOTABLE_COMPUTE_TYPE") or ("float16" if on_cuda else "int8")
        output_dir = env.get("QUOTABLE_OUTPUT_DIR")

        return cls(
            output_dir=Path(output_dir).expanduser() if output_dir else _default_output_dir(env),
            model=model,
            device=device,
            compute_type=compute_type,
            keep_audio=env.get("QUOTABLE_KEEP_AUDIO", "").strip().lower() in _TRUTHY,
            hf_token=env.get("QUOTABLE_HF_TOKEN") or env.get("HF_TOKEN") or None,
        )
