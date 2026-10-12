from pathlib import Path

import pytest

from quotable.transcribe import transcribe_iter

pytestmark = pytest.mark.integration

JFK = Path(__file__).parents[1] / "fixtures" / "jfk.flac"


def test_tiny_transcribes_the_jfk_clip_with_word_timestamps():
    with transcribe_iter(JFK) as run:
        segments = list(run)

    text = " ".join(s.text for s in segments).lower()
    assert "ask not what your country" in text
    assert run.language == "en"
    assert run.duration == pytest.approx(11.0, abs=0.1)
    words = [w for s in segments for w in s.words]
    assert words
    assert all(w.start < w.end for w in words)


def test_resume_only_returns_speech_after_the_resume_point():
    with transcribe_iter(JFK, language="en", start_s=6.0) as run:
        segments = list(run)

    assert segments
    assert all(s.start >= 6.0 - 0.05 for s in segments)
    text = " ".join(s.text for s in segments).lower()
    assert "ask not" not in text
