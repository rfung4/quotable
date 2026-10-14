import io
import threading
from pathlib import Path

import pytest

from quotable import diarize as dz
from quotable import feeds, pipeline
from quotable import transcribe as tx
from quotable.audio import YdlLogger, file_key
from quotable.config import Settings
from quotable.diarize import Turn
from quotable.models import (
    CancelledError,
    DiarizationUnavailableError,
    Method,
    NoCaptionsError,
    Segment,
    SourceError,
    SourceKind,
    TranscriptionError,
    Word,
)
from quotable.store import Store

FIXTURES = Path(__file__).parents[1] / "fixtures"
YT_URL = "https://www.youtube.com/watch?v=abc123"
VTT_URL = "https://captions.example/en.vtt"
FEED_URL = "https://feeds.example/show.xml"
YT_INFO = {
    "extractor_key": "Youtube",
    "id": "abc123",
    "title": "Inaugural",
    "duration": 6.0,
    "webpage_url": YT_URL,
    "subtitles": {"en": [{"ext": "vtt", "url": VTT_URL}]},
    "chapters": [{"start_time": 0, "end_time": 3, "title": "Opening"}, {"start_time": 3}],
}
SEGMENTS = [
    Segment(
        start=0.0,
        end=2.0,
        text="Ask not",
        words=[Word(start=0.0, end=1.0, text="Ask"), Word(start=1.0, end=2.0, text="not")],
    ),
    Segment(
        start=2.0,
        end=4.0,
        text="what your country",
        words=[Word(start=2.0, end=3.0, text="what"), Word(start=3.0, end=4.0, text="country")],
    ),
    Segment(start=4.0, end=6.0, text="can do", words=[Word(start=4.0, end=6.0, text="can do")]),
]


# --- fakes ------------------------------------------------------------------------------


class FakeYDL:
    """A probed YouTube video: metadata, caption bodies by URL, and a recorded download."""

    def __init__(self, info):
        self.info = info
        self.params = {"logger": YdlLogger()}
        self.bodies = {VTT_URL: (FIXTURES / "manual.vtt").read_bytes()}
        self.outdirs: list[Path] = []
        self.opened: list[str] = []
        self.downloads = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def extract_info(self, url, download=True):
        assert download is False, "probing must not download"
        return self.info

    def urlopen(self, url):
        self.opened.append(url)
        return io.BytesIO(self.bodies[url])

    def process_ie_result(self, info, download=True):
        self.downloads += 1
        path = self.outdirs[-1] / "audio.webm"
        path.write_bytes(b"downloaded audio")
        return {"requested_downloads": [{"filepath": str(path)}]}


class FakeNet:
    """Replaces the feed module's two I/O helpers; everything above them runs for real."""

    def __init__(self, bodies):
        self.bodies = bodies
        self.fetched: list[str] = []
        self.streamed: list[str] = []

    def fetch(self, url, max_bytes, *, truncate=False):
        self.fetched.append(url)
        return self.bodies[url]

    def stream_to(self, url, path):
        self.streamed.append(url)
        path.write_bytes(b"enclosure audio")


class FakeWhisper:
    """Stands in for ``transcribe_iter``: replays SEGMENTS from ``start_s``, records each run."""

    def __init__(self, segments=SEGMENTS, fail_after=None):
        self.segments = segments
        self.fail_after = fail_after
        self.calls: list[dict] = []
        self.yielded = 0
        self.gate: threading.Event | None = None  # when set, runs wait on it before decoding

    def __call__(self, audio, language=None, start_s=0.0, settings=None):
        self.calls.append(
            {"audio": Path(audio), "exists": Path(audio).is_file(), "start_s": start_s}
        )
        return FakeRun(self, start_s)


class FakeRun:
    def __init__(self, whisper, start_s):
        self.whisper, self.start_s = whisper, start_s
        self.language = self.duration = None
        self.exited = False

    def __enter__(self):
        self.language, self.duration = "en", 6.0
        return self

    def __exit__(self, *exc):
        self.exited = True

    def __iter__(self):
        if self.whisper.gate is not None:
            assert self.whisper.gate.wait(5)
        for n, seg in enumerate(s for s in self.whisper.segments if s.start >= self.start_s):
            if self.whisper.fail_after is not None and n >= self.whisper.fail_after:
                raise TranscriptionError("Whisper failed mid-run: boom")
            self.whisper.yielded += 1
            yield seg


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(output_dir=tmp_path / "out", model="tiny", device="cpu", compute_type="int8")


@pytest.fixture
def store(settings) -> Store:
    return Store(settings.output_dir)


@pytest.fixture
def net(monkeypatch):
    fake = FakeNet(
        {
            FEED_URL: (FIXTURES / "feed.xml").read_bytes(),
            "https://media.example/ep1.srt": (FIXTURES / "episode_transcript.srt").read_bytes(),
            YT_URL: b"<!doctype html><html><body>video page</body></html>",
        }
    )
    monkeypatch.setattr(feeds, "_fetch", fake.fetch)
    monkeypatch.setattr(feeds, "_stream_to", fake.stream_to)
    return fake


def _youtube(monkeypatch, info) -> FakeYDL:
    ydl = FakeYDL(info)

    def factory(outdir):
        ydl.outdirs.append(outdir)
        return ydl

    monkeypatch.setattr(pipeline, "_youtube_dl", factory)
    return ydl


@pytest.fixture
def youtube(monkeypatch, net) -> FakeYDL:
    return _youtube(monkeypatch, dict(YT_INFO))


@pytest.fixture
def uncaptioned(monkeypatch, net) -> FakeYDL:
    return _youtube(monkeypatch, {k: v for k, v in YT_INFO.items() if k != "subtitles"})


@pytest.fixture
def whisper(monkeypatch) -> FakeWhisper:
    fake = FakeWhisper()
    monkeypatch.setattr(tx, "transcribe_iter", fake)
    return fake


@pytest.fixture
def diarizer(monkeypatch):
    calls = {"pipeline": 0, "diarize": 0}

    def get_pipeline(token):
        calls["pipeline"] += 1

    def diarize(audio, settings):
        calls["diarize"] += 1
        return [Turn(0.0, 3.0, "SPEAKER_00"), Turn(3.0, 6.0, "SPEAKER_01")]

    monkeypatch.setattr(dz, "get_pipeline", get_pipeline)
    monkeypatch.setattr(dz, "diarize", diarize)
    return calls


@pytest.fixture
def recording(tmp_path) -> Path:
    path = tmp_path / "meeting.m4a"
    path.write_bytes(b"not really audio")
    return path


# --- argument and source checks first ---------------------------------------------------


def test_unknown_mode_is_rejected(settings, whisper):
    with pytest.raises(ValueError, match="mode must be one of"):
        pipeline.transcribe("x", mode="fast", settings=settings)


def test_diarize_with_captions_mode_is_a_contradiction(settings, whisper):
    with pytest.raises(ValueError, match="captions carry no speakers"):
        pipeline.transcribe("x", mode="captions", diarize=True, settings=settings)


@pytest.mark.parametrize("source", ["/no/such/file.mp3", "ftp://host/a.mp3", "file:///etc/passwd"])
def test_neither_a_file_nor_an_http_url_is_a_source_error(source, settings, net, whisper):
    with pytest.raises(SourceError, match="not an http"):
        pipeline.transcribe(source, settings=settings)
    assert net.fetched == [] and whisper.calls == []


def test_episode_on_a_non_feed_source_is_refused(settings, recording, youtube, whisper):
    with pytest.raises(SourceError, match="only to podcast RSS feeds"):
        pipeline.transcribe(str(recording), episode=2, settings=settings)
    with pytest.raises(SourceError, match="only to podcast RSS feeds"):
        pipeline.transcribe(YT_URL, episode="ep-2", settings=settings)
    assert youtube.downloads == 0 and whisper.calls == []


# --- captions first ---------------------------------------------------------------------


def test_captions_hit_skips_whisper_and_never_downloads(settings, store, youtube, whisper):
    out = pipeline.transcribe(YT_URL, settings=settings)

    t = out.transcript
    assert (t.method, t.variant, t.language) == (Method.CAPTIONS, "captions-en-captions", "en")
    assert t.id == "Youtube-abc123:captions-en-captions"
    assert t.segments[0].text == "And so, my fellow Americans:"
    assert [c.title for c in t.chapters] == ["Opening", "Chapter 2"]
    assert (t.title, t.duration, t.cite_url) == ("Inaugural", 6.0, "https://youtu.be/abc123?t={t}")
    assert out.cached is False and out.path == store.variant_dir(t.source_key, t.variant)
    assert store.load(t.id) == t
    assert youtube.opened == [VTT_URL]
    assert youtube.downloads == 0 and whisper.calls == []


def test_captions_miss_falls_back_to_whisper(settings, uncaptioned, whisper, capsys):
    out = pipeline.transcribe(YT_URL, settings=settings)

    assert out.transcript.method is Method.WHISPER
    assert out.transcript.variant == "whisper-und-tiny"
    assert out.transcript.language == "en"  # detected, while the variant keys on the request
    assert out.transcript.segments == SEGMENTS
    assert uncaptioned.downloads == 1 and whisper.calls[0]["exists"]
    assert "Transcribing the audio instead" in capsys.readouterr().err


def test_captions_mode_miss_raises_and_caches_nothing(settings, store, uncaptioned, whisper):
    with pytest.raises(NoCaptionsError):
        pipeline.transcribe(YT_URL, mode="captions", settings=settings)
    assert uncaptioned.downloads == 0 and whisper.calls == []
    assert not (store.root / "Youtube-abc123").exists()


def test_unfetchable_captions_fall_back_in_auto_mode(settings, youtube, whisper):
    youtube.bodies[VTT_URL] = b"WEBVTT\n\n"  # an empty track is a miss, never a result

    out = pipeline.transcribe(YT_URL, settings=settings)

    assert out.transcript.method is Method.WHISPER and youtube.downloads == 1


def test_whisper_mode_never_looks_for_captions(settings, youtube, whisper):
    out = pipeline.transcribe(YT_URL, mode="whisper", language="en", settings=settings)

    assert out.transcript.variant == "whisper-en-tiny"
    assert youtube.opened == [] and youtube.downloads == 1


# --- cache ------------------------------------------------------------------------------


def test_cache_hit_does_no_work(settings, youtube, whisper):
    first = pipeline.transcribe(YT_URL, settings=settings)
    second = pipeline.transcribe(YT_URL, settings=settings)

    assert second.cached is True and second.transcript == first.transcript
    assert second.path == first.path
    assert youtube.opened == [VTT_URL]  # fetched once
    assert youtube.downloads == 0 and whisper.calls == []


def test_force_refetches_captions(settings, youtube, whisper):
    pipeline.transcribe(YT_URL, settings=settings)
    forced = pipeline.transcribe(YT_URL, force=True, settings=settings)

    assert forced.cached is False and youtube.opened == [VTT_URL, VTT_URL]


def test_whisper_cache_hit_does_no_work(settings, uncaptioned, whisper):
    pipeline.transcribe(YT_URL, settings=settings)
    again = pipeline.transcribe(YT_URL, settings=settings)

    assert again.cached is True
    assert uncaptioned.downloads == 1 and len(whisper.calls) == 1


def test_force_bypasses_the_cache(settings, recording, whisper):
    pipeline.transcribe(str(recording), settings=settings)
    forced = pipeline.transcribe(str(recording), force=True, settings=settings)

    assert forced.cached is False and len(whisper.calls) == 2
    assert whisper.calls[1]["start_s"] == 0.0


def test_variants_of_one_source_coexist(settings, store, youtube, whisper):
    captioned = pipeline.transcribe(YT_URL, settings=settings)
    whispered = pipeline.transcribe(YT_URL, mode="whisper", settings=settings)

    assert captioned.transcript.source_key == whispered.transcript.source_key
    assert store.load(captioned.transcript.id).method is Method.CAPTIONS
    assert store.load(whispered.transcript.id).method is Method.WHISPER


def test_empty_whisper_output_is_an_error_and_not_cached(settings, store, recording, whisper):
    whisper.segments = []
    with pytest.raises(TranscriptionError, match="No speech"):
        pipeline.transcribe(str(recording), settings=settings)
    assert store.load_variant(file_key(recording), "whisper-und-tiny") is None


# --- local files ------------------------------------------------------------------------


def test_local_file_never_queries_captions(settings, recording, net, youtube, whisper, capsys):
    out = pipeline.transcribe(str(recording), settings=settings)

    t = out.transcript
    assert t.method is Method.WHISPER and t.source_key.startswith("file-")
    assert t.title == "meeting.m4a" and t.cite_url == recording.resolve().as_uri() + "#t={t}"
    assert t.duration == 6.0  # from the Whisper run
    assert whisper.calls[0]["audio"] == recording.resolve()
    assert net.fetched == [] and youtube.outdirs == []
    assert "Transcribing the audio instead" not in capsys.readouterr().err


def test_captions_mode_on_a_local_file_raises(settings, recording, whisper):
    with pytest.raises(NoCaptionsError, match="Local files"):
        pipeline.transcribe(str(recording), mode="captions", settings=settings)
    assert whisper.calls == []


# --- feeds ------------------------------------------------------------------------------


def test_feed_transcript_tag_skips_whisper(settings, net, whisper):
    out = pipeline.transcribe(FEED_URL, episode="Pilot", settings=settings)

    t = out.transcript
    assert t.method is Method.CAPTIONS and t.source_key.startswith("rss-")
    assert t.title == "Episode 1: Pilot"
    assert t.segments[0].text == "Welcome to the pilot episode."
    assert t.cite_url == "https://media.example/ep1.mp3#t={t}"
    assert net.streamed == [] and whisper.calls == []


def test_feed_episode_without_transcript_downloads_the_enclosure(settings, net, whisper):
    out = pipeline.transcribe(FEED_URL, settings=settings)  # default: newest, Episode 3

    t = out.transcript
    assert t.method is Method.WHISPER and t.title == "Episode 3: Q&A"
    assert net.streamed == ["https://media.example/ep3.m4a?token=abc"]
    assert whisper.calls[0]["exists"] and whisper.calls[0]["audio"].name == "audio.m4a"


# --- diarization ------------------------------------------------------------------------


def test_diarize_skips_captions_and_labels_speakers(settings, youtube, whisper, diarizer):
    out = pipeline.transcribe(YT_URL, diarize=True, settings=settings)

    t = out.transcript
    assert t.method is Method.WHISPER and t.variant == "whisper-und-tiny-diarized"
    assert youtube.opened == [] and youtube.downloads == 1
    assert [s.speaker for s in t.segments] == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"]
    assert all(w.speaker for s in t.segments for w in s.words)
    assert t.speakers == ["SPEAKER_00", "SPEAKER_01"]
    assert diarizer == {"pipeline": 1, "diarize": 1}


def test_missing_diarize_extra_fails_before_download_or_whisper(
    monkeypatch, settings, store, youtube, whisper
):
    def unavailable(token):
        raise DiarizationUnavailableError("not installed")

    monkeypatch.setattr(dz, "get_pipeline", unavailable)
    with pytest.raises(DiarizationUnavailableError):
        pipeline.transcribe(YT_URL, diarize=True, settings=settings)
    assert youtube.downloads == 0 and whisper.calls == []


# --- streaming, cancel, resume ----------------------------------------------------------


def test_on_segment_is_called_per_segment_in_order(settings, recording, whisper):
    seen = []
    pipeline.transcribe(
        str(recording), settings=settings, on_segment=lambda *args: seen.append(args)
    )
    assert seen == [(s, s.end, 6.0) for s in SEGMENTS]


def test_cancel_mid_run_keeps_partial_and_the_next_call_resumes(
    settings, store, recording, whisper
):
    cancel = threading.Event()
    key = file_key(recording)

    with pytest.raises(CancelledError):
        pipeline.transcribe(
            str(recording),
            settings=settings,
            cancel=cancel,
            on_segment=lambda seg, done, total: cancel.set(),
        )
    assert whisper.yielded == 1  # stopped before decoding another segment
    assert store.partial_segments(key, "whisper-und-tiny") == SEGMENTS[:1]
    assert store.load_variant(key, "whisper-und-tiny") is None

    out = pipeline.transcribe(str(recording), settings=settings)

    assert whisper.calls[1]["start_s"] == SEGMENTS[0].end
    assert out.transcript.segments == SEGMENTS  # no repeats across the resume
    assert store.partial_segments(key, "whisper-und-tiny") == []


def test_cancel_set_before_the_run_downloads_nothing(settings, uncaptioned, whisper):
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(CancelledError):
        pipeline.transcribe(YT_URL, settings=settings, cancel=cancel)
    assert uncaptioned.downloads == 0 and whisper.calls == []


def test_force_discards_a_partial_run(settings, store, recording, whisper):
    whisper.fail_after = 1
    with pytest.raises(TranscriptionError):
        pipeline.transcribe(str(recording), settings=settings)
    whisper.fail_after = None

    pipeline.transcribe(str(recording), force=True, settings=settings)

    assert whisper.calls[1]["start_s"] == 0.0


def test_second_concurrent_caller_waits_then_hits_the_cache(settings, recording, whisper):
    whisper.gate = threading.Event()
    results: dict[str, object] = {}

    def call(name):
        try:
            results[name] = pipeline.transcribe(str(recording), settings=settings)
        except BaseException as exc:  # surfaced by the asserts below
            results[name] = exc

    first = threading.Thread(target=call, args=("first",))
    first.start()
    for _ in range(500):  # until the first caller holds the lock and has started Whisper
        if whisper.calls:
            break
        threading.Event().wait(0.01)
    second = threading.Thread(target=call, args=("second",))
    second.start()
    second.join(0.3)
    assert second.is_alive(), "the second caller should be waiting on the variant lock"

    whisper.gate.set()
    first.join(5)
    second.join(5)

    assert results["first"].cached is False
    assert results["second"].cached is True
    assert results["second"].transcript == results["first"].transcript
    assert len(whisper.calls) == 1


# --- temp audio -------------------------------------------------------------------------


def test_temp_audio_is_removed_on_success(settings, uncaptioned, whisper):
    pipeline.transcribe(YT_URL, settings=settings)

    assert whisper.calls[0]["exists"]
    assert not uncaptioned.outdirs[0].exists()


def test_temp_audio_is_removed_on_failure_and_the_partial_kept(
    settings, store, uncaptioned, whisper
):
    whisper.fail_after = 1
    with pytest.raises(TranscriptionError):
        pipeline.transcribe(YT_URL, settings=settings)

    assert not uncaptioned.outdirs[0].exists()
    assert store.partial_segments("Youtube-abc123", "whisper-und-tiny") == SEGMENTS[:1]


def test_keep_audio_leaves_the_download(settings, uncaptioned, whisper, capsys):
    kept = Settings(**{**settings.__dict__, "keep_audio": True})
    pipeline.transcribe(YT_URL, settings=kept)

    assert (uncaptioned.outdirs[0] / "audio.webm").is_file()
    assert str(uncaptioned.outdirs[0]) in capsys.readouterr().err


# --- media_info -------------------------------------------------------------------------


def test_media_info_for_a_local_file_reads_its_duration():
    info = pipeline.media_info(str(FIXTURES / "jfk.flac"))

    assert info.kind is SourceKind.FILE and info.title == "jfk.flac"
    assert 10.5 < info.duration < 11.5
    assert info.has_captions is False and info.cite_url.endswith("jfk.flac#t={t}")


def test_media_info_for_a_url_downloads_nothing(youtube):
    youtube.info = {
        **YT_INFO,
        "subtitles": {"en": YT_INFO["subtitles"]["en"], "live_chat": [{"ext": "json"}]},
        "automatic_captions": {"fr-orig": [{"ext": "vtt", "url": "https://captions.example/f"}]},
    }
    info = pipeline.media_info(YT_URL)

    assert info.kind is SourceKind.URL and info.title == "Inaugural" and info.duration == 6.0
    assert info.has_captions is True and info.languages == ["en"]
    assert [c.title for c in info.chapters] == ["Opening", "Chapter 2"]
    assert youtube.opened == [] and youtube.downloads == 0


def test_media_info_lists_the_original_auto_caption_language(uncaptioned):
    uncaptioned.info = {
        **uncaptioned.info,
        "automatic_captions": {"de-orig": [{"ext": "vtt", "url": "https://captions.example/d"}]},
    }
    assert pipeline.media_info(YT_URL).languages == ["de"]


def test_media_info_for_a_feed_names_the_show_unless_an_episode_is_asked_for(net):
    show = pipeline.media_info(FEED_URL)
    pilot = pipeline.media_info(FEED_URL, episode="Pilot")
    first = pipeline.media_info(FEED_URL, episode=1)

    # No episode: the title is the show's; the other fields still describe the newest one.
    assert show.kind is SourceKind.FEED and show.title == "Quotable Test Cast"
    assert show.has_captions is False and show.duration == 3723.0
    assert [e.title for e in show.episodes][:2] == ["Episode 3: Q&A", "Episode 2: Interviews"]
    assert pilot.has_captions is True and pilot.title == "Episode 1: Pilot"
    assert first.title == "Episode 3: Q&A"
    assert net.streamed == []


def test_media_info_lists_each_recent_episodes_declared_transcript_types(net):
    listed = {e.title: e for e in pipeline.media_info(FEED_URL).episodes}
    assert "text/html" in listed["Episode 2: Interviews"].transcript_types
    assert listed["Episode 3: Q&A"].transcript_types == []


def test_media_info_refuses_episode_on_a_file():
    with pytest.raises(SourceError, match="only to podcast RSS feeds"):
        pipeline.media_info(str(FIXTURES / "jfk.flac"), episode=1)
