import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from quotable import feeds
from quotable.models import DownloadError, NoCaptionsError, SourceError, SourceKind

FIXTURES = Path(__file__).parents[1] / "fixtures"
FEED_URL = "https://feeds.example/show.xml"


class FakeResp:
    def __init__(self, body: bytes, url: str) -> None:
        self._body, self._url, self._pos, self.closed = body, url, 0, False

    def read(self, n: int = -1) -> bytes:
        end = len(self._body) if n < 0 else self._pos + n
        chunk, self._pos = self._body[self._pos : end], min(end, len(self._body))
        return chunk

    def geturl(self) -> str:
        return self._url

    def close(self) -> None:
        self.closed = True


class FakeOpener:
    """Replaces the module opener: serves bodies by URL, can fake a redirect target."""

    def __init__(self, bodies=None, final_url=None, error=None):
        self.bodies, self.final_url, self.error = bodies or {}, final_url or {}, error
        self.opened: list[str] = []

    def open(self, req, timeout=None):
        assert timeout == feeds.TIMEOUT_S
        self.opened.append(req.full_url)
        if self.error:
            raise self.error
        return FakeResp(self.bodies[req.full_url], self.final_url.get(req.full_url, req.full_url))


@pytest.fixture
def net(monkeypatch):
    opener = FakeOpener({FEED_URL: (FIXTURES / "feed.xml").read_bytes()})
    monkeypatch.setattr(feeds, "_OPENER", opener)
    return opener


def _feed(net) -> feeds.Feed:
    return feeds.load_feed(FEED_URL)


def _hostile_episode(transcript_url=None, enclosure_url="https://media.example/x.mp3"):
    refs = (feeds.TranscriptRef(url=transcript_url, kind="srt"),) if transcript_url else ()
    return feeds.Episode(
        index=1,
        title="Hostile",
        guid="h",
        published=None,
        duration=None,
        enclosure_url=enclosure_url,
        transcripts=refs,
    )


# --- deny cases first -----------------------------------------------------------------


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://media.example/a.srt", "data:,x"])
def test_non_http_transcript_url_is_refused_without_opening(net, url):
    with pytest.raises(SourceError, match="non-http"):
        feeds.fetch_transcript(_hostile_episode(transcript_url=url))
    assert net.opened == []


@pytest.mark.parametrize("url", ["file:///home/u/.ssh/id_rsa", "ftp://media.example/a.mp3"])
def test_non_http_enclosure_is_refused_without_opening_or_writing(net, tmp_path, url):
    with pytest.raises(SourceError, match="non-http"):
        feeds.download_enclosure(_hostile_episode(enclosure_url=url), tmp_path)
    assert net.opened == []
    assert list(tmp_path.iterdir()) == []


def test_final_url_off_http_is_refused(monkeypatch):
    url = "https://media.example/ep.srt"
    opener = FakeOpener(
        {url: b"1\n00:00:00,000 --> 00:00:01,000\nx\n"}, {url: "file:///etc/passwd"}
    )
    monkeypatch.setattr(feeds, "_OPENER", opener)
    with pytest.raises(SourceError):
        feeds.fetch_transcript(_hostile_episode(transcript_url=url))


def test_oversize_transcript_is_rejected(monkeypatch):
    url = "https://media.example/huge.srt"
    body = b"x" * (feeds.TRANSCRIPT_MAX_BYTES + 1)
    monkeypatch.setattr(feeds, "_OPENER", FakeOpener({url: body}))
    with pytest.raises(SourceError, match="exceeds"):
        feeds.fetch_transcript(_hostile_episode(transcript_url=url))


def test_module_opener_has_no_file_ftp_or_data_handler():
    for scheme in ("file", "ftp", "data"):
        assert not any(hasattr(h, f"{scheme}_open") for h in feeds._OPENER.handlers)


class _Server(BaseHTTPRequestHandler):
    routes: dict[str, tuple[int, dict, bytes]] = {}

    def do_GET(self):
        status, headers, body = self.routes[self.path]
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    _Server.routes = {}
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Server)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", _Server.routes
    httpd.shutdown()
    httpd.server_close()


@pytest.mark.parametrize(
    "location", ["file:///etc/passwd", "ftp://127.0.0.1/a.srt", "data:text/plain,hi"]
)
def test_real_opener_refuses_redirect_off_http(server, location):
    base, routes = server
    routes["/t.srt"] = (302, {"Location": location}, b"")
    with pytest.raises(SourceError, match="non-http"):
        feeds.fetch_transcript(_hostile_episode(transcript_url=f"{base}/t.srt"))


def test_real_opener_follows_http_redirects_and_reads(server):
    base, routes = server
    routes["/old.srt"] = (301, {"Location": "/new.srt"}, b"")
    routes["/new.srt"] = (200, {}, (FIXTURES / "episode_transcript.srt").read_bytes())
    segs = feeds.fetch_transcript(_hostile_episode(transcript_url=f"{base}/old.srt"))
    assert segs[0].text == "Welcome to the pilot episode."


# --- sniffing -------------------------------------------------------------------------


def test_sniff_true_for_rss_reading_only_64kb(monkeypatch):
    big = (FIXTURES / "feed.xml").read_bytes() + b"<!--" + b"x" * 200_000 + b"-->"
    opener = FakeOpener({FEED_URL: big})
    monkeypatch.setattr(feeds, "_OPENER", opener)
    assert feeds.is_feed(FEED_URL) is True


@pytest.mark.parametrize(
    "body",
    [
        b"<!doctype html><html><head><title>x</title></head></html>",
        b'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>',
        b"\x00\x01 not xml",
        b"",
    ],
    ids=["html", "atom", "binary", "empty"],
)
def test_sniff_false_for_non_rss(monkeypatch, body):
    monkeypatch.setattr(feeds, "_OPENER", FakeOpener({FEED_URL: body}))
    assert feeds.is_feed(FEED_URL) is False


def test_sniff_false_on_network_error_and_for_non_http(monkeypatch):
    opener = FakeOpener(error=OSError("unreachable"))
    monkeypatch.setattr(feeds, "_OPENER", opener)
    assert feeds.is_feed(FEED_URL) is False
    assert feeds.is_feed("file:///tmp/feed.xml") is False
    assert opener.opened == [FEED_URL]


# --- parsing and episode choice -------------------------------------------------------


def test_episodes_newest_first_with_metadata(net):
    feed = _feed(net)
    assert feed.title == "Quotable Test Cast"
    assert [(e.index, e.title) for e in feed.episodes] == [
        (1, "Episode 3: Q&A"),
        (2, "Episode 2: Interviews"),
        (3, "Episode 1: Pilot"),
        (4, "Bonus"),
    ]
    ep3, ep2, ep1, bonus = feed.episodes
    assert ep3.duration == 3723.0 and ep1.duration == 2730.0 and ep2.duration == 1800.0
    assert ep3.published.startswith("2026-10-12") and bonus.published is None
    assert [t.kind for t in ep2.transcripts] == ["vtt", "srt", "json"]  # text/html ignored


def test_list_episodes_reports_has_transcript(net):
    infos = feeds.list_episodes(FEED_URL)
    assert [(i.index, i.guid, i.has_transcript) for i in infos] == [
        (1, "ep-3", False),
        (2, "ep-2", True),
        (3, "1", True),
        (4, None, False),
    ]


def test_pick_episode_by_guid_index_title_and_default(net):
    feed = _feed(net)
    assert feeds.pick_episode(feed).guid == "ep-3"
    assert feeds.pick_episode(feed, "ep-2").guid == "ep-2"
    assert feeds.pick_episode(feed, "1").guid == "1"  # a digit guid wins over index 1
    assert feeds.pick_episode(feed, 1).guid == "ep-3"  # an int is always an index
    assert feeds.pick_episode(feed, "4").title == "Bonus"
    assert feeds.pick_episode(feed, "interview").guid == "ep-2"


def test_pick_episode_miss_lists_recent_titles(net):
    feed = _feed(net)
    with pytest.raises(SourceError) as exc:
        feeds.pick_episode(feed, "nonexistent")
    assert "1: Episode 3: Q&A" in exc.value.fix
    with pytest.raises(SourceError, match="out of range"):
        feeds.pick_episode(feed, 9)


def test_not_rss_and_bad_xml_are_source_errors():
    with pytest.raises(SourceError, match="Not an RSS"):
        feeds.parse_feed(b"<feed/>", FEED_URL)
    with pytest.raises(SourceError, match="valid XML"):
        feeds.parse_feed(b"<rss><channel>", FEED_URL)


def test_feed_fetch_error_is_a_source_error(monkeypatch):
    monkeypatch.setattr(feeds, "_OPENER", FakeOpener(error=OSError("HTTP 404")))
    with pytest.raises(SourceError, match="fetch the feed"):
        feeds.load_feed(FEED_URL)


# --- transcripts ----------------------------------------------------------------------


def test_transcript_preference_json_then_srt_then_vtt(net):
    ep2 = feeds.pick_episode(_feed(net), "ep-2")
    assert feeds.choose_transcript(ep2).kind == "json"
    ep1 = feeds.pick_episode(_feed(net), "1")
    assert feeds.choose_transcript(ep1).kind == "srt"
    assert feeds.choose_transcript(feeds.pick_episode(_feed(net))) is None


def test_requested_language_outranks_format():
    ep = _hostile_episode()
    ep = feeds.Episode(
        **{
            **ep.__dict__,
            "transcripts": (
                feeds.TranscriptRef("https://m.example/a.json", "json", "de"),
                feeds.TranscriptRef("https://m.example/a.srt", "srt", "en"),
            ),
        }
    )
    assert feeds.choose_transcript(ep, "en-US").kind == "srt"
    assert feeds.choose_transcript(ep).kind == "json"


def test_transcript_tag_is_used_and_the_enclosure_is_not_touched(net):
    net.bodies["https://media.example/ep1.srt"] = (FIXTURES / "episode_transcript.srt").read_bytes()
    segs = feeds.fetch_transcript(feeds.pick_episode(_feed(net), "1"))
    assert [s.text for s in segs] == [
        "Welcome to the pilot episode.",
        "Today: why transcripts should be citable.",
        "Bye.",
    ]
    assert not any("ep1.mp3" in u for u in net.opened)


def test_no_transcript_returns_none_so_the_enclosure_is_used(net, tmp_path):
    net.bodies["https://media.example/ep3.m4a?token=abc"] = b"audio bytes"
    ep = feeds.pick_episode(_feed(net))
    assert feeds.fetch_transcript(ep) is None
    src = feeds.download_enclosure(ep, tmp_path)
    assert src.kind is SourceKind.FEED
    assert src.path == tmp_path / "audio.m4a" and src.path.read_bytes() == b"audio bytes"
    assert src.source_key == feeds.episode_key(ep) and src.source_key.startswith("rss-")
    assert src.cite_url == "https://media.example/ep3.m4a?token=abc#t={t}"
    assert (src.title, src.duration) == ("Episode 3: Q&A", 3723.0)


def test_enclosure_network_error_is_a_download_error(monkeypatch, tmp_path):
    monkeypatch.setattr(feeds, "_OPENER", FakeOpener(error=OSError("reset")))
    with pytest.raises(DownloadError):
        feeds.download_enclosure(_hostile_episode(), tmp_path)


def test_episode_key_prefers_guid_then_enclosure():
    a = _hostile_episode()
    no_guid = feeds.Episode(**{**a.__dict__, "guid": None})
    assert feeds.episode_key(a) != feeds.episode_key(no_guid)
    with pytest.raises(SourceError):
        feeds.episode_key(feeds.Episode(**{**a.__dict__, "guid": None, "enclosure_url": None}))


def test_srt_parse():
    segs = feeds.parse_srt((FIXTURES / "episode_transcript.srt").read_text())
    assert [(s.start, s.end) for s in segs] == [(0.5, 3.25), (3.25, 6.0), (3600.0, 3601.5)]


@pytest.mark.parametrize(
    "body, exc",
    [(b"", "empty"), (b"{not json", "parsed"), (b'{"segments": [{"body": " "}]}', "empty")],
    ids=["empty-srt", "bad-json", "blank-json"],
)
def test_bad_transcripts_are_a_miss(monkeypatch, body, exc):
    url = "https://media.example/t.json" if b"{" in body else "https://media.example/t.srt"
    kind = "json" if b"{" in body else "srt"
    ep = feeds.Episode(
        **{**_hostile_episode().__dict__, "transcripts": (feeds.TranscriptRef(url, kind),)}
    )
    monkeypatch.setattr(feeds, "_OPENER", FakeOpener({url: body}))
    with pytest.raises(NoCaptionsError, match=exc):
        feeds.fetch_transcript(ep)


def test_transcript_fetch_error_is_a_miss(monkeypatch):
    monkeypatch.setattr(feeds, "_OPENER", FakeOpener(error=OSError("HTTP 500")))
    with pytest.raises(NoCaptionsError, match="fetch"):
        feeds.fetch_transcript(_hostile_episode(transcript_url="https://media.example/t.srt"))


def test_json_phrase_level_keeps_speakers():
    body = json.dumps(
        {
            "version": "1.0.0",
            "segments": [
                {"speaker": "Alice", "startTime": 0.5, "endTime": 2.0, "body": "Hello there."},
                {"speaker": "Bob", "startTime": 2.0, "endTime": 3.5, "body": "Hi, Alice."},
            ],
        }
    )
    segs = feeds.parse_json_transcript(body)
    assert [(s.speaker, s.text, s.words) for s in segs] == [
        ("Alice", "Hello there.", []),
        ("Bob", "Hi, Alice.", []),
    ]


def test_json_word_level_is_regrouped_into_sentences_with_words():
    words = [
        ("Alice", 0.0, 0.3, "Hello"),
        ("Alice", 0.3, 0.6, "there."),
        ("Alice", 0.7, 0.9, "How"),
        ("Alice", 0.9, 1.1, "are"),
        ("Bob", 1.2, 1.5, "Fine"),  # speaker change splits
        ("Bob", 3.0, 3.2, "thanks"),  # >1 s gap splits
    ]
    body = json.dumps(
        {
            "segments": [
                {"speaker": s, "startTime": a, "endTime": b, "body": t} for s, a, b, t in words
            ]
        }
    )
    segs = feeds.parse_json_transcript(body)
    assert [(s.speaker, s.text, s.start, s.end) for s in segs] == [
        ("Alice", "Hello there.", 0.0, 0.6),
        ("Alice", "How are", 0.7, 1.1),
        ("Bob", "Fine", 1.2, 1.5),
        ("Bob", "thanks", 3.0, 3.2),
    ]
    assert [w.text for w in segs[0].words] == ["Hello", "there."]
    assert segs[0].words[1].speaker == "Alice"


def test_duration_formats():
    assert feeds.parse_duration("90") == 90.0
    assert feeds.parse_duration("1:30") == 90.0
    assert feeds.parse_duration("1:00:00") == 3600.0
    assert feeds.parse_duration("1:2:3:4") is None
    assert feeds.parse_duration("soon") is None
    assert feeds.parse_duration(None) is None


def test_transcript_tag_under_the_github_namespace_url_is_detected():
    # The Podcasting 2.0 show's own feed declares the original namespace URL.
    body = b"""<?xml version="1.0"?>
<rss version="2.0"
     xmlns:podcast="https://github.com/Podcastindex-org/podcast-namespace/blob/main/docs/1.0.md">
<channel><title>PC20</title><item><title>Ep</title><guid>g1</guid>
<enclosure url="https://media.example/ep.mp3" type="audio/mpeg"/>
<podcast:transcript url="https://media.example/ep.srt" type="application/srt"/>
</item></channel></rss>"""
    feed = feeds.parse_feed(body, FEED_URL)
    assert [t.kind for t in feed.episodes[0].transcripts] == ["srt"]
