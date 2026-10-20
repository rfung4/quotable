import json
import subprocess
import sys
from pathlib import Path

import pytest

from quotable import cli, daemon, pipeline, server, shim
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


@pytest.fixture
def serve_targets(monkeypatch):
    ran = []
    monkeypatch.setattr(server, "run", lambda: ran.append("in-process"))
    monkeypatch.setattr(shim, "main", lambda: ran.append("shim") or 0)
    return ran


@pytest.mark.parametrize("value, target", [(None, "shim"), ("on", "shim"), (" ON ", "shim")])
def test_serve_uses_the_shim_on_posix(monkeypatch, serve_targets, value, target):
    if value is None:
        monkeypatch.delenv("QUOTABLE_DAEMON", raising=False)
    else:
        monkeypatch.setenv("QUOTABLE_DAEMON", value)
    assert cli.main(["serve"]) == 0
    assert serve_targets == [target]


def test_serve_runs_the_server_in_process_when_off(monkeypatch, serve_targets):
    monkeypatch.setenv("QUOTABLE_DAEMON", "off")
    assert cli.main(["serve"]) == 0
    assert serve_targets == ["in-process"]


def test_serve_runs_in_process_on_windows(monkeypatch, serve_targets):
    monkeypatch.setenv("QUOTABLE_DAEMON", "on")
    monkeypatch.setattr(cli, "_shim_supported", lambda: False)
    assert cli.main(["serve"]) == 0
    assert serve_targets == ["in-process"]


def test_serve_rejects_an_unknown_daemon_mode(monkeypatch, serve_targets, capsys):
    monkeypatch.setenv("QUOTABLE_DAEMON", "maybe")
    assert cli.main(["serve"]) == 2
    assert serve_targets == []
    assert "QUOTABLE_DAEMON='maybe' is invalid; use on or off" in capsys.readouterr().err


def test_serve_help_still_works(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["serve", "--help"])
    assert exc.value.code == 0 and "usage: quotable serve" in capsys.readouterr().out


def test_daemon_command_runs_the_shared_process(monkeypatch):
    ran = []
    monkeypatch.setattr(daemon, "run", lambda: ran.append(True))
    assert cli.main(["daemon"]) == 0
    assert ran == [True]


def test_serve_through_the_shim_imports_only_the_stdlib(tmp_path):
    # stdin at EOF: the shim exits before connecting, so nothing is spawned.
    probe = (
        "import sys; before = set(sys.modules); from quotable import cli; "
        "code = cli.main(['serve']); "
        "new = {m.split('.')[0] for m in set(sys.modules) - before}; "
        "print(code, sorted(new - set(sys.stdlib_module_names) - {'quotable'}), "
        "'importlib.metadata' in sys.modules)"
    )
    env = {"PATH": "/usr/bin:/bin", "QUOTABLE_DAEMON": "on", "XDG_STATE_HOME": str(tmp_path)}
    out = subprocess.run(
        [sys.executable, "-c", probe],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert out.strip() == "0 [] False"


def test_unknown_format_is_rejected(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["get", "a.flac", "--format", "docx"])
    assert exc.value.code == 2
