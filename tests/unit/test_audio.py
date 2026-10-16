from pathlib import Path

import pytest
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError as YtDlpDownloadError

from quotable import audio
from quotable.audio import YdlLogger
from quotable.models import DownloadError, SourceError, SourceKind

YT_INFO = {"extractor_key": "Youtube", "id": "dQw4w9WgXcQ", "title": "T", "duration": 212.0}


class FakeYDL:
    def __init__(self, info=None, error=None, result=None, warnings=()):
        self.info = info
        self.error = error
        self.result = result
        self.params = {"logger": YdlLogger(list(warnings))}
        self.calls: list[tuple] = []

    def extract_info(self, url, download=True):
        self.calls.append(("extract_info", url, download))
        if self.error:
            raise self.error
        return self.info

    def process_ie_result(self, info, download=True):
        self.calls.append(("process_ie_result", info.get("id"), download))
        if self.error:
            raise self.error
        return self.result


# --- source keys ----------------------------------------------------------------------


def test_file_key_is_a_content_hash(tmp_path):
    a = tmp_path / "a.flac"
    a.write_bytes(b"same bytes")
    moved = tmp_path / "sub" / "renamed.m4a"
    moved.parent.mkdir()
    moved.write_bytes(b"same bytes")
    other = tmp_path / "b.flac"
    other.write_bytes(b"different")

    key = audio.file_key(a)
    assert key.startswith("file-") and len(key) == len("file-") + 16
    assert audio.file_key(moved) == key
    assert audio.file_key(other) != key


def test_url_key_uses_extractor_and_id():
    assert audio.url_key(YT_INFO) == "Youtube-dQw4w9WgXcQ"


@pytest.mark.parametrize("bad_id", ["../../etc", "a:b", "x" * 65, "ünï"])
def test_url_key_hashes_unsafe_ids(bad_id):
    key = audio.url_key({"extractor_key": "Generic", "id": bad_id})
    assert key.startswith("Generic-")
    assert key.removeprefix("Generic-").isalnum() and len(key) == len("Generic-") + 16


def test_url_key_without_id_is_an_error():
    with pytest.raises(SourceError):
        audio.url_key({"extractor_key": "Generic"})


# --- local passthrough ----------------------------------------------------------------


def test_local_path_detection(tmp_path):
    f = tmp_path / "talk.mp3"
    f.write_bytes(b"x")
    assert audio.local_path(str(f)) == f.resolve()
    assert audio.local_path(str(tmp_path)) is None  # a directory is not a source
    assert audio.local_path(str(tmp_path / "missing.mp3")) is None
    assert audio.local_path("https://example.com/talk.mp3") is None


def test_local_source_passes_the_file_through(tmp_path):
    f = tmp_path / "talk.mp3"
    f.write_bytes(b"x")
    src = audio.local_source(f)
    assert src.kind is SourceKind.FILE
    assert src.path == f
    assert src.source_key == audio.file_key(f)
    assert src.cite_url == f.resolve().as_uri() + "#t={t}"


# --- options --------------------------------------------------------------------------


def test_options_download_bestaudio_without_postprocessing(tmp_path):
    opts = audio.ydl_options(tmp_path)
    assert opts["format"] == "bestaudio/best"
    assert opts["noplaylist"] is True and opts["quiet"] is True and opts["noprogress"] is True
    assert "postprocessors" not in opts
    assert opts["outtmpl"].startswith(str(tmp_path))
    assert set(opts["js_runtimes"]) == {"deno", "node"}


def test_real_youtubedl_accepts_the_options(tmp_path):
    with YoutubeDL(audio.ydl_options(tmp_path)) as ydl:
        assert ydl.params["format"] == "bestaudio/best"


# --- cite urls ------------------------------------------------------------------------


def test_cite_url_per_kind(tmp_path):
    assert audio.url_cite(YT_INFO, "https://www.youtube.com/watch?v=dQw4w9WgXcQ") == (
        "https://youtu.be/dQw4w9WgXcQ?t={t}"
    )
    generic = {"extractor_key": "Vimeo", "id": "1", "webpage_url": "https://vimeo.com/1#old"}
    assert audio.url_cite(generic, "https://vimeo.com/1") == "https://vimeo.com/1#t={t}"
    assert audio.url_cite({"id": "1"}, "https://x.example/a.mp3") == "https://x.example/a.mp3#t={t}"
    assert audio.fragment_cite("https://feed.example/ep.mp3?u=1#x") == (
        "https://feed.example/ep.mp3?u=1#t={t}"
    )
    assert audio.file_cite(tmp_path / "a b.wav").endswith("/a%20b.wav#t={t}")


# --- probe / download -----------------------------------------------------------------


def test_probe_does_not_download():
    ydl = FakeYDL(info=YT_INFO)
    assert audio.probe(ydl, "https://youtu.be/dQw4w9WgXcQ") is YT_INFO
    assert ydl.calls == [("extract_info", "https://youtu.be/dQw4w9WgXcQ", False)]


def test_probe_rejects_playlists():
    ydl = FakeYDL(info={"_type": "playlist", "id": "PL1", "entries": []})
    with pytest.raises(SourceError, match="Playlists"):
        audio.probe(ydl, "https://youtube.com/playlist?list=PL1")


def test_download_returns_the_downloaded_file(tmp_path):
    f = tmp_path / "audio.webm"
    f.write_bytes(b"opus")
    ydl = FakeYDL(result={**YT_INFO, "requested_downloads": [{"filepath": str(f)}]})
    src = audio.download(ydl, YT_INFO, "https://youtu.be/dQw4w9WgXcQ")
    assert ydl.calls == [("process_ie_result", "dQw4w9WgXcQ", True)]
    assert (src.kind, src.path, src.source_key) == (SourceKind.URL, f, "Youtube-dQw4w9WgXcQ")
    assert (src.title, src.duration) == ("T", 212.0)
    assert src.cite_url == "https://youtu.be/dQw4w9WgXcQ?t={t}"


def test_download_without_a_file_is_an_error(tmp_path):
    ydl = FakeYDL(result={**YT_INFO, "requested_downloads": [{"filepath": str(tmp_path / "x")}]})
    with pytest.raises(DownloadError, match="without producing"):
        audio.download(ydl, YT_INFO, "u")


# --- error mapping --------------------------------------------------------------------


def test_ytdlp_download_error_maps_to_ours():
    ydl = FakeYDL(error=YtDlpDownloadError("ERROR: [youtube] x: Video unavailable"))
    with pytest.raises(DownloadError) as exc:
        audio.probe(ydl, "https://youtu.be/x")
    assert str(exc.value) == "[youtube] x: Video unavailable"
    assert "yt-dlp" in exc.value.fix


def test_unsupported_url_is_a_source_error():
    ydl = FakeYDL(error=YtDlpDownloadError("ERROR: Unsupported URL: https://example.com/"))
    with pytest.raises(SourceError) as exc:
        audio.probe(ydl, "https://example.com/")
    assert "media URL" in exc.value.fix


def test_js_runtime_failure_names_deno_and_node_even_when_only_warned():
    # yt-dlp reports the missing runtime as a warning, then raises a generic format error.
    ydl = FakeYDL(
        error=YtDlpDownloadError("ERROR: [youtube] x: Requested format is not available"),
        warnings=["n challenge solving failed: Ensure you have a supported JavaScript runtime"],
    )
    with pytest.raises(DownloadError) as exc:
        audio.download(ydl, YT_INFO, "https://youtu.be/x")
    assert "Deno" in exc.value.fix and "Node" in exc.value.fix


def test_logger_keeps_warnings_and_writes_to_stderr(capsys):
    logger = YdlLogger()
    logger.info("noise")
    logger.warning("careful")
    out = capsys.readouterr()
    assert logger.warnings == ["careful"]
    assert out.out == "" and "careful" in out.err


def test_is_url():
    assert audio.is_url("https://a.example/x") and audio.is_url("http://a.example")
    assert not audio.is_url("file:///etc/passwd")
    assert not audio.is_url(str(Path("relative/file.mp3")))


def test_generic_urls_with_the_same_file_name_get_different_keys():
    a = {
        "extractor_key": "Generic",
        "id": "master",
        "webpage_url": "https://a.example/s/master.m3u8",
    }
    b = {**a, "webpage_url": "https://b.example/live/master.m3u8"}
    assert audio.url_key(a) != audio.url_key(b)
    assert audio.url_key(a).startswith("Generic-")


@pytest.mark.parametrize("live", [{"is_live": True}, {"live_status": "is_upcoming"}])
def test_probe_rejects_live_streams(live):
    with pytest.raises(SourceError, match="Live streams"):
        audio.probe(FakeYDL(info={**YT_INFO, **live}), "https://youtu.be/dQw4w9WgXcQ")
