"""Real pyannote on the JFK clip. Needs the diarize extra and an HF token with the
model's terms accepted. Under ``-m diarize`` a missing token *fails*: a skipped check
would read as a pass."""

import os
from pathlib import Path

import pytest

from quotable.config import Settings
from quotable.diarize import TERMS_URL, assign_speakers, diarize
from quotable.transcribe import transcribe_iter

pytestmark = pytest.mark.diarize

JFK = Path(__file__).parents[1] / "fixtures" / "jfk.flac"


def test_jfk_clip_is_one_speaker_on_every_word():
    token = os.environ.get("QUOTABLE_HF_TOKEN") or os.environ.get("HF_TOKEN")
    assert token, f"Set QUOTABLE_HF_TOKEN after accepting the terms at {TERMS_URL}"
    settings = Settings.from_env()

    with transcribe_iter(JFK, settings=settings) as run:
        segments = list(run)
    turns = diarize(JFK, settings)
    labelled = assign_speakers(segments, turns)

    words = [w for s in labelled for w in s.words]
    assert words and all(w.speaker for w in words)
    assert len({w.speaker for w in words}) == 1
    assert all(s.speaker for s in labelled)
