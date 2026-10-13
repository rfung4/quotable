import json
from pathlib import Path

import pytest

from quotable import cli, pipeline, server
from quotable.models import (
    MediaInfo,
    Method,
    NoCaptionsError,
    Segment,
    SourceKind,
    Transcript,
)

SEGMENTS = [
    Segment(start=0.0, end=2.0, text="Ask not.", speaker="SPEAKER_00"),
    Segment(start=2.0, end=4.5, text="What your country can do."),
]


def _outcome() -> pipeline.Outcome:
    transcript = Transcript(
        source_key="file-abc", variant="whisper-en-tiny", method=Method.WHISPER, segments=SEGMENTS
    )
    return pipeline.Outcome(transcript=transcript, path=Path("/tmp/x"), cached=False)


@pytest.fixture
def calls(monkeypatch):
    seen: dict = {}

    def fake(source, **kwargs):
        seen["source"], seen["kwargs"] = source, kwargs
        for seg in SEGMENTS:
            kwargs["on_segment"](seg, seg.end, 4.5)
        return _outcome()

    monkeypatch.setattr(pipeline, "transcribe", fake)
    return seen


def test_get_txt_stdout_only_transcript_and_streams_segments_to_stderr(calls, capsys):
    assert cli.main(["get", "jfk.flac"]) == 0
    out, err = capsys.readouterr()
    assert out == "SPEAKER_00: Ask not.\nWhat your country can do.\n"
    assert "[00:00:00] SPEAKER_00: Ask not." in err
    assert "What your country can do." in err
    assert calls["source"] == "jfk.flac"


def test_get_srt_format(calls, capsys):
    assert cli.main(["get", "jfk.flac", "--format", "srt"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("1\n00:00:00,000 --> 00:00:02,000\n")


def test_get_json_format_carries_segments(calls, capsys):
    assert cli.main(["get", "jfk.flac", "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [s["text"] for s in data["segments"]] == ["Ask not.", "What your country can do."]


def test_get_passes_options_through(calls):
    argv = ["get", "https://x/feed", "--mode", "whisper", "--lang", "en", "--diarize"]
    argv += ["--episode", "42", "--force"]
    assert cli.main(argv) == 0
    kw = calls["kwargs"]
    assert (kw["mode"], kw["language"], kw["diarize"], kw["episode"], kw["force"]) == (
        "whisper",
        "en",
        True,
        "42",
        True,
    )


def test_get_defaults(calls):
    cli.main(["get", "jfk.flac"])
    kw = calls["kwargs"]
    assert (kw["mode"], kw["language"], kw["diarize"], kw["episode"], kw["force"]) == (
        "auto",
        None,
        False,
        None,
        False,
    )


def test_info_prints_media_info_json(monkeypatch, capsys):
    got = {}

    def fake(source, episode=None):
        got["args"] = (source, episode)
        return MediaInfo(source=source, kind=SourceKind.FILE, title="Clip", duration=11.0)

    monkeypatch.setattr(pipeline, "media_info", fake)
    assert cli.main(["info", "jfk.flac", "--episode", "3"]) == 0
    assert got["args"] == ("jfk.flac", "3")
    assert json.loads(capsys.readouterr().out)["title"] == "Clip"


def test_typed_error_is_one_stderr_line_with_fix_and_exit_1(monkeypatch, capsys):
    def boom(source, **kwargs):
        raise NoCaptionsError("No captions.", fix="Use --mode whisper.")

    monkeypatch.setattr(pipeline, "transcribe", boom)
    assert cli.main(["get", "https://x/v"]) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert err == "quotable: No captions. Fix: Use --mode whisper.\n"


def test_value_error_exits_1(monkeypatch, capsys):
    def boom(source, **kwargs):
        raise ValueError("diarize=True needs the audio")

    monkeypatch.setattr(pipeline, "transcribe", boom)
    assert cli.main(["get", "a.flac", "--diarize", "--mode", "captions"]) == 1
    assert "diarize=True needs the audio" in capsys.readouterr().err


def test_serve_runs_the_server(monkeypatch):
    ran = []
    monkeypatch.setattr(server, "run", lambda: ran.append(True))
    assert cli.main(["serve"]) == 0
    assert ran == [True]


def test_unknown_format_is_rejected(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["get", "a.flac", "--format", "docx"])
    assert exc.value.code == 2
