import json
import threading
import time
from pathlib import Path

import anyio
import pytest
from mcp.client import Client
from mcp.types import ResourceLink

from quotable import pipeline, server
from quotable.config import Settings
from quotable.models import (
    CancelledError,
    Chapter,
    MediaInfo,
    Method,
    NoCaptionsError,
    Segment,
    SourceKind,
    Transcript,
    Word,
)
from quotable.server import UNTRUSTED, TranscribeResult
from quotable.store import Store, render_timestamped, render_txt

SOURCE = "/recordings/meeting.m4a"


def _segments(n: int) -> list[Segment]:
    return [
        Segment(
            start=i * 10.0,
            end=i * 10.0 + 9.5,
            text=f"Line {i:04d} " + "word " * 18,
            speaker=f"SPEAKER_0{i % 2}",
            words=[Word(start=i * 10.0, end=i * 10.0 + 1, text=f"Line{i}")],
        )
        for i in range(n)
    ]


class FakePipeline:
    """Replaces ``pipeline.transcribe``: emits segments every ``delay`` s, honours cancel."""

    def __init__(self, store: Store, segments: list[Segment], delay: float = 0.0, error=None):
        self.store, self.segments, self.delay, self.error = store, segments, delay, error
        self.calls: list[dict] = []
        self.cancel: threading.Event | None = None
        self.emitted = 0
        self.stopped = threading.Event()  # set when the fake leaves its loop for any reason
        self.cached = False
        self.waits = 0  # times to report waiting behind another run before starting

    def __call__(
        self, source, *, on_segment=None, on_wait=None, cancel=None, settings=None, **kwargs
    ):
        self.calls.append({"source": source, **kwargs})
        self.cancel = cancel
        try:
            if self.error:
                raise self.error
            for _ in range(self.waits):
                on_wait()
            for seg in self.segments:
                if self.delay:
                    time.sleep(self.delay)
                if cancel.is_set():
                    raise CancelledError("Cancelled between segments.")
                on_segment(seg, seg.end, self.segments[-1].end)
                self.emitted += 1
        finally:
            self.stopped.set()
        t = Transcript(
            source_key="file-0123456789abcdef",
            variant="whisper-und-tiny-diarized",
            method=Method.WHISPER,
            language="en",
            segments=self.segments,
            chapters=[Chapter(title="Intro", start=0.0, end=60.0)],
            title="meeting.m4a",
            duration=self.segments[-1].end,
            cite_url="file:///recordings/meeting.m4a#t={t}",
        )
        return pipeline.Outcome(t, self.store.finalize(t), cached=self.cached)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(output_dir=tmp_path / "out", model="tiny", device="cpu", compute_type="int8")


@pytest.fixture
def store(settings) -> Store:
    return Store(settings.output_dir)


@pytest.fixture
def fake(monkeypatch, store) -> FakePipeline:
    fake = FakePipeline(store, _segments(300))
    monkeypatch.setattr(pipeline, "transcribe", fake)
    return fake


def _run(settings, scenario, **client_kwargs):
    async def main():
        with anyio.fail_after(10):
            async with Client(server.create_server(settings), **client_kwargs) as client:
                return await scenario(client)

    return anyio.run(main)


# --- surface ----------------------------------------------------------------------------


def test_tools_schemas_and_annotations(settings):
    async def scenario(client):
        return {t.name: t for t in (await client.list_tools()).tools}

    tools = _run(settings, scenario)

    assert sorted(tools) == ["media_info", "read_transcript", "transcribe"]
    tx, rd, mi = tools["transcribe"], tools["read_transcript"], tools["media_info"]

    assert tx.input_schema["required"] == ["source"]
    assert set(tx.input_schema["properties"]) == {
        "source",
        "mode",
        "language",
        "diarize",
        "episode",
        "force",
    }
    assert tx.input_schema["properties"]["mode"]["enum"] == ["auto", "captions", "whisper"]
    assert set(tx.output_schema["properties"]) == set(TranscribeResult.model_fields)
    assert set(rd.input_schema["properties"]) == {
        "id",
        "offset",
        "limit",
        "start_s",
        "end_s",
        "timestamps",
    }
    assert rd.input_schema["properties"]["offset"]["minimum"] == 0
    assert set(rd.output_schema["properties"]) >= {"notice", "id", "text", "next_offset"}
    assert set(mi.output_schema["properties"]) == set(MediaInfo.model_fields)

    a = tx.annotations
    assert (a.read_only_hint, a.destructive_hint, a.idempotent_hint, a.open_world_hint) == (
        False,
        False,
        True,
        True,
    )
    for tool in (rd, mi):
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.idempotent_hint is True
    assert rd.annotations.open_world_hint is False
    assert "read_transcript" in tx.description and "cached" in tx.description.lower()


def test_instructions_are_short_and_name_the_workflow(settings):
    text = server.INSTRUCTIONS
    assert len(text.splitlines()) <= 5
    assert "transcribe" in text and "read_transcript" in text


# --- transcribe -------------------------------------------------------------------------


def test_transcribe_returns_structured_result_text_and_resource_link(settings, fake):
    async def scenario(client):
        return await client.call_tool(
            "transcribe", {"source": SOURCE, "diarize": True, "episode": None}
        )

    result = _run(settings, scenario)

    assert not result.is_error
    data = TranscribeResult.model_validate(result.structured_content)
    assert data.id == "file-0123456789abcdef:whisper-und-tiny-diarized"
    assert data.method is Method.WHISPER and data.cached is False
    assert data.speakers == ["SPEAKER_00", "SPEAKER_01"]
    assert data.chapters == [Chapter(title="Intro", start=0.0, end=60.0)]
    assert data.cite_url == "file:///recordings/meeting.m4a#t={t}"
    assert Path(data.path, "transcript.json").is_file()
    assert data.next_offset is not None and len(data.text) <= 20_000
    assert render_txt(fake.segments).startswith(data.text)

    # A client that forwards only the structured result still shows the notice first.
    assert next(iter(result.structured_content)) == "notice"
    assert data.notice == UNTRUSTED

    text, link = result.content
    assert text.text.startswith(UNTRUSTED + "\n")
    assert data.id in text.text and data.text in text.text
    assert f"offset={data.next_offset}" in text.text
    assert isinstance(link, ResourceLink) and str(link.uri) == f"transcript://{data.id}"
    assert fake.calls == [
        {
            "source": SOURCE,
            "mode": "auto",
            "language": None,
            "diarize": True,
            "episode": None,
            "force": False,
        }
    ]


def test_a_cached_result_says_so(settings, fake):
    fake.cached = True

    async def scenario(client):
        return await client.call_tool("transcribe", {"source": SOURCE})

    result = _run(settings, scenario)
    assert result.structured_content["cached"] is True
    assert "cached: yes" in result.content[0].text


def test_progress_notifications_carry_the_newest_line(settings, fake):
    fake.segments = _segments(3)
    seen = []

    async def on_progress(progress, total, message):
        seen.append((progress, total, message))

    async def scenario(client):
        return await client.call_tool(
            "transcribe", {"source": SOURCE}, progress_callback=on_progress
        )

    _run(settings, scenario)

    assert [p for p, _, _ in seen] == [9.5, 19.5, 29.5]
    assert all(total == 29.5 for _, total, _ in seen)
    assert [m for _, _, m in seen] == [s.text[: server.PROGRESS_CHARS] for s in fake.segments]


def test_queue_wait_is_reported_as_progress_that_never_goes_back(settings, fake):
    fake.segments = _segments(2)
    fake.waits = 1
    seen = []

    async def on_progress(progress, total, message):
        seen.append((progress, total, message))

    async def scenario(client):
        return await client.call_tool(
            "transcribe", {"source": SOURCE}, progress_callback=on_progress
        )

    _run(settings, scenario)

    assert seen[0] == (0.0, None, server.QUEUE_MESSAGE)
    assert [p for p, _, _ in seen[1:]] == [9.5, 19.5]


def test_a_typed_failure_is_a_tool_error_with_the_fix(settings, fake):
    fake.error = NoCaptionsError("No caption track for this source.")

    async def scenario(client):
        return await client.call_tool("transcribe", {"source": SOURCE, "mode": "captions"})

    result = _run(settings, scenario)

    assert result.is_error
    text = result.content[0].text
    assert "No caption track for this source." in text
    assert "Fix: Retry with mode='auto' or mode='whisper'" in text


def test_an_argument_error_from_the_pipeline_reaches_the_agent(settings, fake):
    fake.error = ValueError("diarize=True needs the audio (captions carry no speakers)")

    async def scenario(client):
        return await client.call_tool("transcribe", {"source": SOURCE})

    result = _run(settings, scenario)
    assert result.is_error and "captions carry no speakers" in result.content[0].text


def test_an_invalid_mode_is_rejected_by_the_schema(settings, fake):
    async def scenario(client):
        return await client.call_tool("transcribe", {"source": SOURCE, "mode": "fast"})

    result = _run(settings, scenario)
    assert result.is_error and fake.calls == []


def test_cancel_sets_the_pipeline_event_and_stops_between_segments(settings, fake):
    fake.segments, fake.delay = _segments(100), 0.05
    first_progress = anyio.Event()

    async def on_progress(progress, total, message):
        first_progress.set()

    async def scenario(client):
        async with anyio.create_task_group() as tg:

            async def call():
                await client.call_tool(
                    "transcribe", {"source": SOURCE}, progress_callback=on_progress
                )

            tg.start_soon(call)
            await first_progress.wait()
            tg.cancel_scope.cancel()
        cancelled_at = time.monotonic()
        seen = await anyio.to_thread.run_sync(lambda: fake.cancel.wait(0.5))
        latency = time.monotonic() - cancelled_at
        stopped = await anyio.to_thread.run_sync(lambda: fake.stopped.wait(1))
        return seen, latency, stopped

    seen, latency, stopped = _run(settings, scenario)

    assert seen and latency < 0.5
    assert stopped and 1 <= fake.emitted < len(fake.segments)


# --- read_transcript and resources ------------------------------------------------------


def _cached_transcript(store, n=300) -> Transcript:
    t = Transcript(
        source_key="Youtube-abc123",
        variant="captions-en-captions",
        method=Method.CAPTIONS,
        language="en",
        segments=_segments(n),
        title="Talk",
    )
    store.finalize(t)
    return t


def test_paging_by_offset_round_trips_the_whole_text(settings, store):
    t = _cached_transcript(store)

    async def scenario(client):
        pages, offset = [], 0
        while offset is not None:
            r = await client.call_tool(
                "read_transcript", {"id": t.id, "offset": offset, "limit": 5000}
            )
            assert not r.is_error
            assert r.content[0].text.startswith(UNTRUSTED + "\n")
            assert list(r.structured_content)[:2] == ["notice", "id"]
            assert r.structured_content["notice"] == UNTRUSTED
            pages.append(r.structured_content["text"])
            offset = r.structured_content["next_offset"]
        return pages

    pages = _run(settings, scenario)

    assert len(pages) > 1 and "".join(pages) == render_txt(t.segments)


def test_paging_by_time_window_with_timestamps(settings, store):
    t = _cached_transcript(store)

    async def scenario(client):
        return await client.call_tool(
            "read_transcript", {"id": t.id, "start_s": 100, "end_s": 130, "timestamps": True}
        )

    r = _run(settings, scenario)

    assert r.structured_content["text"] == render_timestamped(t.segments[10:13])
    assert r.structured_content["text"].startswith("[00:01:40] SPEAKER_00: Line 0010")
    assert r.structured_content["next_offset"] is None
    assert "End of transcript." in r.content[0].text


def test_read_transcript_refuses_a_hostile_id(settings, store):
    async def scenario(client):
        return await client.call_tool("read_transcript", {"id": "../../etc:passwd"})

    r = _run(settings, scenario)
    assert r.is_error and "Unknown transcript id" in r.content[0].text


def test_resources_serve_text_and_meta(settings, store):
    t = _cached_transcript(store, n=3)

    async def scenario(client):
        templates = (await client.list_resource_templates()).resource_templates
        text = await client.read_resource(f"transcript://{t.id}")
        meta = await client.read_resource(f"transcript://{t.id}/meta")
        return templates, text.contents[0], meta.contents[0]

    templates, text, meta = _run(settings, scenario)

    assert {r.uri_template: r.mime_type for r in templates} == {
        "transcript://{id}": "text/plain",
        "transcript://{id}/meta": "application/json",
    }
    assert text.mime_type == "text/plain" and text.text == render_timestamped(t.segments)
    data = json.loads(meta.text)
    assert data["id"] == t.id and data["segment_count"] == 3
    assert data["speakers"] == ["SPEAKER_00", "SPEAKER_01"] and "segments" not in data


def test_an_unknown_resource_is_a_not_found_error(settings, store):
    async def scenario(client):
        with pytest.raises(Exception, match="call `transcribe` first"):
            await client.read_resource("transcript://Youtube-zzz:captions-en-captions")

    _run(settings, scenario)


# --- media_info -------------------------------------------------------------------------


def test_media_info_returns_the_model(settings, monkeypatch):
    info = MediaInfo(source=SOURCE, kind=SourceKind.FILE, title="meeting.m4a", duration=61.0)
    calls = []
    monkeypatch.setattr(pipeline, "media_info", lambda *a: calls.append(a) or info)

    async def scenario(client):
        return await client.call_tool("media_info", {"source": SOURCE, "episode": 2})

    r = _run(settings, scenario)

    assert not r.is_error and MediaInfo.model_validate(r.structured_content) == info
    assert calls == [(SOURCE, 2)]
