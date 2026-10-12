from pathlib import Path

import pytest

from quotable.config import Settings
from quotable.models import (
    CancelledError,
    NoCaptionsError,
    QuotableError,
    Segment,
    Transcript,
)


def _no_gpu() -> int:
    return 0


def _one_gpu() -> int:
    return 1


def test_cpu_defaults_when_no_gpu():
    s = Settings.from_env({"XDG_DATA_HOME": "/data"}, cuda_count=_no_gpu)
    assert (s.device, s.model, s.compute_type) == ("cpu", "base", "int8")
    assert s.output_dir == Path("/data/quotable")
    assert s.keep_audio is False and s.hf_token is None


def test_cuda_defaults_when_gpu_present():
    s = Settings.from_env({}, cuda_count=_one_gpu)
    assert (s.device, s.model, s.compute_type) == ("cuda", "turbo", "float16")


def test_env_overrides():
    env = {
        "QUOTABLE_DEVICE": "CPU",
        "QUOTABLE_MODEL": "tiny",
        "QUOTABLE_COMPUTE_TYPE": "float32",
        "QUOTABLE_OUTPUT_DIR": "/tmp/q",
        "QUOTABLE_KEEP_AUDIO": "1",
        "HF_TOKEN": "hf_plain",
    }
    s = Settings.from_env(env, cuda_count=_one_gpu)
    assert (s.device, s.model, s.compute_type) == ("cpu", "tiny", "float32")
    assert s.output_dir == Path("/tmp/q")
    assert s.keep_audio is True
    assert s.hf_token == "hf_plain"


def test_quotable_token_wins_over_hf_token_and_is_not_in_repr():
    s = Settings.from_env(
        {"QUOTABLE_HF_TOKEN": "hf_mine", "HF_TOKEN": "hf_other", "QUOTABLE_DEVICE": "cpu"}
    )
    assert s.hf_token == "hf_mine"
    assert "hf_mine" not in repr(s)


def test_xdg_data_home_unset_falls_back_to_home(monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: Path("/home/u"))
    s = Settings.from_env({"QUOTABLE_DEVICE": "cpu"})
    assert s.output_dir == Path("/home/u/.local/share/quotable")


def test_invalid_device_rejected():
    with pytest.raises(ValueError, match="QUOTABLE_DEVICE"):
        Settings.from_env({"QUOTABLE_DEVICE": "tpu"})


def test_errors_carry_fix_hint_and_override():
    err = NoCaptionsError("no captions")
    assert isinstance(err, QuotableError) and err.fix
    assert CancelledError("stopped", fix="resume").fix == "resume"


def test_transcript_id_and_speakers_in_first_seen_order():
    t = Transcript(
        source_key="file-abc",
        variant="whisper-en-base-diarized",
        method="whisper",
        segments=[
            Segment(start=0, end=1, text="a", speaker="B"),
            Segment(start=1, end=2, text="b", speaker="A"),
            Segment(start=2, end=3, text="c", speaker="B"),
            Segment(start=3, end=4, text="d"),
        ],
    )
    assert t.id == "file-abc:whisper-en-base-diarized"
    assert t.speakers == ["B", "A"]
