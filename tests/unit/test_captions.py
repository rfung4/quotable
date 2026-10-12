from pathlib import Path

import pytest

from quotable.captions import chapters_from_info, fetch, parse_json3, parse_vtt, pick_track
from quotable.models import NoCaptionsError

FIXTURES = Path(__file__).parents[1] / "fixtures"


def _fmts(*exts: str, lang: str = "x") -> list[dict]:
    return [{"ext": ext, "url": f"https://cap.example/{lang}.{ext}"} for ext in exts]


def _yt(**kw) -> dict:
    return {"extractor_key": "Youtube", **kw}


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body


class FakeYDL:
    """Stands in for ``YoutubeDL``: serves bodies by URL and records what was opened."""

    def __init__(self, bodies: dict[str, bytes] | None = None, error: Exception | None = None):
        self.bodies = bodies or {}
        self.error = error
        self.opened: list[str] = []

    def urlopen(self, req: str) -> FakeResponse:
        self.opened.append(req)
        if self.error:
            raise self.error
        return FakeResponse(self.bodies[req])


# --- track choice ---------------------------------------------------------------------


def test_manual_beats_automatic_for_the_requested_language():
    info = _yt(
        subtitles={"en": _fmts("vtt", "json3", lang="en-manual")},
        automatic_captions={"en": _fmts("json3", lang="en-auto")},
    )
    track = pick_track(info, "en")
    assert (track.language, track.auto, track.ext) == ("en", False, "json3")


def test_exact_language_then_base_language_then_automatic():
    info = _yt(
        subtitles={"en": _fmts("vtt"), "en-GB": _fmts("vtt")},
        automatic_captions={"de": _fmts("json3")},
    )
    assert pick_track(info, "en-GB").language == "en-GB"
    assert pick_track(info, "en-AU").language == "en"
    de = pick_track(info, "de")
    assert (de.language, de.auto) == ("de", True)
    assert pick_track(info, "ja") is None


def test_no_language_prefers_manual_in_the_video_language():
    info = _yt(
        language="fr",
        subtitles={"en": _fmts("vtt"), "fr-CA": _fmts("vtt")},
        automatic_captions={"fr-orig": _fmts("json3")},
    )
    assert pick_track(info).language == "fr-CA"


def test_no_language_uses_the_original_automatic_track_never_a_translation():
    info = _yt(
        automatic_captions={"de": _fmts("json3"), "en-orig": _fmts("json3"), "en": _fmts("json3")}
    )
    track = pick_track(info)
    assert (track.language, track.auto) == ("en", True)
    assert track.url.endswith("x.json3")


def test_no_language_and_only_unlabelled_automatic_tracks_is_a_miss():
    info = _yt(automatic_captions={"de": _fmts("json3"), "fr": _fmts("json3")})
    assert pick_track(info) is None


def test_video_language_finds_automatic_track_without_orig_label():
    info = _yt(language="en", automatic_captions={"de": _fmts("json3"), "en": _fmts("json3")})
    assert pick_track(info).language == "en"


def test_json3_only_for_youtube_otherwise_vtt():
    tracks = {"en": _fmts("json3", "vtt")}
    assert pick_track(_yt(subtitles=tracks)).ext == "json3"
    assert pick_track({"extractor_key": "Vimeo", "subtitles": tracks}).ext == "vtt"


def test_tracks_without_a_supported_format_and_live_chat_are_skipped():
    info = _yt(
        subtitles={"live_chat": _fmts("json"), "en": _fmts("srv3", "ttml"), "es": _fmts("vtt")},
    )
    assert pick_track(info).language == "es"
    assert pick_track(_yt(subtitles={"live_chat": _fmts("json")})) is None


# --- parsing --------------------------------------------------------------------------


def test_json3_automatic_track_yields_words_with_offsets():
    segs = parse_json3((FIXTURES / "autosub.json3").read_text(), words=True)
    assert [s.text for s in segs] == ["ask not what", "your country can do", "for you"]
    first = segs[0]
    assert (first.start, first.end) == (0.12, 2.31)  # display overlap capped at the next event
    assert [(w.text, w.start, w.end) for w in first.words] == [
        ("ask", 0.12, 0.52),
        ("not", 0.52, 0.82),
        ("what", 0.82, 2.31),
    ]
    assert segs[1].words[1].start == 2.6
    assert segs[-1].end == 7.0
    assert all(w.start <= w.end for s in segs for w in s.words)


def test_json3_manual_track_has_no_word_level_timing():
    segs = parse_json3((FIXTURES / "autosub.json3").read_text(), words=False)
    assert segs and all(s.words == [] for s in segs)


def test_rolling_autosub_vtt_collapses_repeats():
    body = (FIXTURES / "rolling_autosub.vtt").read_text()
    assert "\n \n" in body, "fixture lost YouTube's single-space lines (editor stripped them?)"
    segs = parse_vtt(body)
    assert [(s.text, s.start, s.end) for s in segs] == [
        ("ask not what", 0.0, 2.31),
        ("your country can do", 2.32, 4.95),
        ("for you", 4.96, 7.0),
    ]
    texts = [s.text for s in segs]
    assert len(texts) == len(set(texts))
    assert not any("<" in t for t in texts)


def test_manual_vtt_strips_tags_entities_ids_and_notes():
    segs = parse_vtt((FIXTURES / "manual.vtt").read_text())
    assert [s.text for s in segs] == [
        "And so, my fellow Americans:",
        "ask not what your country",
        "can do for you & ask",
        "what you can do for your country.",
    ]
    assert (segs[1].start, segs[1].end) == (3.5, 6.25)  # mm:ss.mmm form
    assert segs[-1].start == 3600.0  # hh:mm:ss.mmm form


# --- fetch ----------------------------------------------------------------------------


def test_fetch_downloads_through_the_ydl_opener_and_parses():
    info = _yt(automatic_captions={"en-orig": _fmts("json3", lang="en")})
    url = "https://cap.example/en.json3"
    ydl = FakeYDL({url: (FIXTURES / "autosub.json3").read_bytes()})
    segs, track = fetch(ydl, info)
    assert ydl.opened == [url]
    assert track.auto and segs[0].words


@pytest.mark.parametrize(
    "body",
    [b"WEBVTT\n\n", b"WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n \n"],
    ids=["no-cues", "blank-cue"],
)
def test_empty_track_raises_no_captions(body):
    info = {"subtitles": {"en": _fmts("vtt", lang="en")}}
    ydl = FakeYDL({"https://cap.example/en.vtt": body})
    with pytest.raises(NoCaptionsError, match="empty"):
        fetch(ydl, info)


def test_unparseable_json3_is_a_miss_not_a_result():
    info = _yt(subtitles={"en": _fmts("json3", lang="en")})
    ydl = FakeYDL({"https://cap.example/en.json3": b"<html>rate limited</html>"})
    with pytest.raises(NoCaptionsError, match="parsed"):
        fetch(ydl, info)


def test_no_track_raises_with_the_requested_language():
    with pytest.raises(NoCaptionsError, match="'ja'"):
        fetch(FakeYDL(), _yt(subtitles={"en": _fmts("vtt")}), "ja")


def test_fetch_error_is_a_miss():
    info = {"subtitles": {"en": _fmts("vtt", lang="en")}}
    with pytest.raises(NoCaptionsError, match="fetch"):
        fetch(FakeYDL(error=OSError("HTTP 429")), info)


def test_non_http_track_url_is_refused_without_opening():
    info = {"subtitles": {"en": [{"ext": "vtt", "url": "file:///etc/passwd"}]}}
    ydl = FakeYDL()
    with pytest.raises(NoCaptionsError, match="http"):
        fetch(ydl, info)
    assert ydl.opened == []


# --- chapters -------------------------------------------------------------------------


def test_chapters_sorted_with_missing_ends_filled():
    info = {
        "duration": 300.0,
        "chapters": [
            {"start_time": 120.0, "title": "Q&amp;A"},
            {"start_time": 0.0, "end_time": 120.0, "title": "Intro"},
            {"start_time": 200.0, "title": ""},
        ],
    }
    assert [(c.title, c.start, c.end) for c in chapters_from_info(info)] == [
        ("Intro", 0.0, 120.0),
        ("Q&A", 120.0, 200.0),
        ("Chapter 3", 200.0, 300.0),
    ]


def test_no_chapters():
    assert chapters_from_info({}) == []
    assert chapters_from_info({"chapters": None}) == []
