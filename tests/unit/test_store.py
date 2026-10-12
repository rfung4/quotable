from pathlib import Path

import pytest

from quotable import store as st
from quotable.models import (
    Chapter,
    Method,
    Segment,
    SourceError,
    Transcript,
    TranscriptionError,
    Word,
)
from quotable.store import Store, parse_id, variant_name

KEY = "Youtube-abc123"
VARIANT = "whisper-en-turbo-diarized"


def _transcript(variant=VARIANT, segments=None) -> Transcript:
    return Transcript(
        source_key=KEY,
        variant=variant,
        method=Method.WHISPER,
        language="en",
        title="Talk",
        duration=3725.0,
        cite_url="https://youtu.be/abc123?t={t}",
        chapters=[Chapter(title="Intro", start=0.0, end=60.0)],
        segments=(
            segments
            if segments is not None
            else [
                Segment(
                    start=0.0,
                    end=2.5,
                    text="Ask not.",
                    speaker="SPEAKER_00",
                    words=[
                        Word(start=0.0, end=1.0, text="Ask", speaker="SPEAKER_00"),
                        Word(start=1.0, end=2.5, text="not.", speaker="SPEAKER_00"),
                    ],
                ),
                Segment(start=61.25, end=63.0, text="What your country", speaker="SPEAKER_01"),
                Segment(start=3723.5, end=3725.0, text="can do."),
            ]
        ),
    )


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(tmp_path / "out")


# --- deny cases first -----------------------------------------------------------------

BAD_IDS = [
    "../x:y",
    "a:../../b",
    "a/b:c",
    "a:.",
    "a:..",
    "/etc/passwd",
    "/abs:path",
    ".locks:x",
    "Youtube-abc:.whisper.work",  # the in-progress work dir
    "nocolon",
    "a:b:c",
    "",
    "a:b\x00",
    "a:b\n",
]


def _outcome(call):
    try:
        call()
    except Exception as exc:  # recorded; asserted after the patches are undone
        return exc
    return None


@pytest.mark.parametrize("bad", BAD_IDS)
def test_bad_ids_rejected_before_any_filesystem_access(store, monkeypatch, bad):
    def no_fs(*a, **k):
        raise AssertionError("filesystem touched")

    # Patched only inside this block: pytest itself needs Path to report a failure.
    with monkeypatch.context() as m:
        for name in ("resolve", "exists", "is_file", "open", "read_text", "mkdir", "stat"):
            m.setattr(Path, name, no_fs)
        load_exc = _outcome(lambda: store.load(bad))
        page_exc = _outcome(lambda: store.page(bad))

    for exc in (load_exc, page_exc):
        assert isinstance(exc, SourceError), f"{bad!r}: {exc!r}"
        assert str(store.root) not in str(exc) and str(store.root) not in exc.fix


def test_symlinked_source_dir_pointing_outside_is_rejected(store, tmp_path):
    outside = tmp_path / "outside" / "v"
    outside.mkdir(parents=True)
    (outside / "transcript.json").write_text(_transcript().model_dump_json())
    store.root.mkdir(parents=True)
    (store.root / "evil").symlink_to(tmp_path / "outside")
    with pytest.raises(SourceError):
        store.load("evil:v")


def test_valid_id_parses():
    assert parse_id("Youtube-abc123:whisper-en-US-large-v3.5") == (
        "Youtube-abc123",
        "whisper-en-US-large-v3.5",
    )


# --- variant names --------------------------------------------------------------------


def test_variant_names_follow_decision_22():
    assert variant_name("captions", "en", "turbo") == "captions-en-captions"
    assert variant_name(Method.WHISPER, None, "turbo") == "whisper-und-turbo"
    assert variant_name("whisper", "pt-BR", "large-v3", diarized=True) == (
        "whisper-pt-BR-large-v3-diarized"
    )
    # a local model path keeps only its directory name, and stays path-safe
    assert variant_name("whisper", "en", "/models/my model") == "whisper-en-my_model"
    for v in ("captions-en-captions", "whisper-en-my_model"):
        parse_id(f"k:{v}")


# --- round trip and renderers ---------------------------------------------------------


def test_round_trip_keeps_words_speakers_and_chapters(store):
    t = _transcript()
    final = store.finalize(t)
    assert sorted(p.name for p in final.iterdir()) == sorted(st.FILES)
    loaded = store.load(f"{KEY}:{VARIANT}")
    assert loaded == t
    assert loaded.segments[0].words[1].speaker == "SPEAKER_00"
    assert loaded.chapters[0].title == "Intro"


def test_srt_golden_output(store):
    store.finalize(_transcript())
    srt = (store.root / KEY / VARIANT / "transcript.srt").read_text()
    assert srt == (
        "1\n00:00:00,000 --> 00:00:02,500\nSPEAKER_00: Ask not.\n"
        "\n"
        "2\n00:01:01,250 --> 00:01:03,000\nSPEAKER_01: What your country\n"
        "\n"
        "3\n01:02:03,500 --> 01:02:05,000\ncan do.\n"
    )
    assert (store.root / KEY / VARIANT / "transcript.txt").read_text() == (
        "SPEAKER_00: Ask not.\nSPEAKER_01: What your country\ncan do.\n"
    )


def test_timestamped_rendering():
    assert st.render_timestamped(_transcript().segments) == (
        "[00:00:00] SPEAKER_00: Ask not.\n"
        "[00:01:01] SPEAKER_01: What your country\n"
        "[01:02:03] can do.\n"
    )


def test_two_variants_of_one_source_coexist(store):
    store.finalize(_transcript("whisper-en-turbo"))
    store.finalize(_transcript("captions-en-captions", [Segment(start=0, end=1, text="cap")]))
    assert store.load(f"{KEY}:whisper-en-turbo").segments[0].text == "Ask not."
    assert store.load(f"{KEY}:captions-en-captions").segments[0].text == "cap"


def test_finalize_replaces_an_existing_entry(store):
    store.finalize(_transcript(segments=[Segment(start=0, end=1, text="old")]))
    store.finalize(_transcript(segments=[Segment(start=0, end=1, text="new")]))
    assert store.load(f"{KEY}:{VARIANT}").segments[0].text == "new"
    assert [p.name for p in (store.root / KEY).iterdir()] == [VARIANT]


def test_missing_and_corrupt_entries_are_misses(store):
    assert store.load(f"{KEY}:{VARIANT}") is None
    d = store.root / KEY / VARIANT
    d.mkdir(parents=True)
    (d / "transcript.json").write_text("{truncated")
    assert store.load(f"{KEY}:{VARIANT}") is None


# --- atomicity ------------------------------------------------------------------------


def test_exception_mid_write_leaves_no_cache_entry(store, monkeypatch):
    def boom(segments):
        raise OSError("disk full")

    monkeypatch.setattr(st, "render_srt", boom)
    with pytest.raises(OSError):
        store.finalize(_transcript())
    assert store.load(f"{KEY}:{VARIANT}") is None
    assert not any(p.name.startswith(f".{VARIANT}.staging") for p in (store.root / KEY).iterdir())


def test_failed_replace_keeps_nothing_half_written(store, monkeypatch):
    store.finalize(_transcript(segments=[Segment(start=0, end=1, text="old")]))
    monkeypatch.setattr(st, "render_txt", lambda segs: (_ for _ in ()).throw(OSError("x")))
    with pytest.raises(OSError):
        store.finalize(_transcript(segments=[Segment(start=0, end=1, text="new")]))
    assert store.load(f"{KEY}:{VARIANT}").segments[0].text == "old"


def test_empty_transcript_is_never_cached(store):
    with pytest.raises(TranscriptionError, match="nothing was cached"):
        store.finalize(_transcript(segments=[]))
    assert store.load(f"{KEY}:{VARIANT}") is None


# --- partial work and resume ----------------------------------------------------------


def test_partial_resume_point_and_cleanup_on_finalize(store):
    assert store.resume_point(KEY, VARIANT) == 0.0
    for seg in _transcript().segments[:2]:
        store.append_partial(KEY, VARIANT, seg)
    assert store.resume_point(KEY, VARIANT) == 63.0
    assert [s.text for s in store.partial_segments(KEY, VARIANT)] == [
        "Ask not.",
        "What your country",
    ]
    assert store.load(f"{KEY}:{VARIANT}") is None  # partial work is not a result
    store.finalize(_transcript())
    assert not (store.root / KEY / f".{VARIANT}.work").exists()


def test_torn_last_line_is_ignored(store):
    store.append_partial(KEY, VARIANT, Segment(start=0, end=4.0, text="whole"))
    with (store.root / KEY / f".{VARIANT}.work" / st.PARTIAL).open("a") as fh:
        fh.write('{"start": 4.0, "end": 9.0, "te')
    assert store.resume_point(KEY, VARIANT) == 4.0


def test_variant_lock_lives_under_locks(store):
    lock = store.variant_lock(KEY, VARIANT)
    with lock:
        assert Path(lock.lock_file).parent == store.root / ".locks"
        assert lock.is_locked


# --- paging ---------------------------------------------------------------------------


def _many(n=10) -> Transcript:
    return _transcript(
        segments=[Segment(start=i * 10.0, end=i * 10.0 + 9, text=f"line {i:02d}") for i in range(n)]
    )


def test_offset_paging_walks_the_whole_text_on_line_boundaries(store):
    store.finalize(_many())
    tid = f"{KEY}:{VARIANT}"
    full = st.render_txt(_many().segments)
    pages, offset = [], 0
    while offset is not None:
        page = store.page(tid, offset=offset, limit=20)
        assert page.text.endswith("\n")
        pages.append(page.text)
        offset = page.next_offset
    assert "".join(pages) == full
    assert len(pages) == 5  # "line NN\n" is 8 chars: two whole lines fit in each 20-char page


def test_exact_fit_and_overlong_line(store):
    store.finalize(_many(2))
    tid = f"{KEY}:{VARIANT}"
    whole = store.page(tid, limit=16)
    assert (whole.text, whole.next_offset, whole.total_chars) == ("line 00\nline 01\n", None, 16)
    cut = store.page(tid, limit=5)  # a line longer than the limit is cut, not skipped
    assert (cut.text, cut.next_offset) == ("line ", 5)
    assert store.page(tid, offset=5, limit=5).text == "00\n"


def test_offset_past_the_end_is_an_empty_last_page(store):
    store.finalize(_many(2))
    page = store.page(f"{KEY}:{VARIANT}", offset=999)
    assert (page.text, page.next_offset) == ("", None)


def test_time_window_selects_overlapping_segments(store):
    store.finalize(_many())
    tid = f"{KEY}:{VARIANT}"
    page = store.page(tid, start_s=25.0, end_s=40.0)
    assert page.text == "line 02\nline 03\n"  # 20-29 overlaps 25; 40-49 starts at end_s
    assert (page.start_s, page.end_s, page.total_chars) == (25.0, 40.0, 16)
    assert store.page(tid, start_s=85.0).text == "line 08\nline 09\n"
    assert store.page(tid, end_s=10.0).text == "line 00\n"


def test_time_window_with_timestamps_and_paging(store):
    store.finalize(_many())
    tid = f"{KEY}:{VARIANT}"
    first = store.page(tid, start_s=60.0, limit=20, timestamps=True)
    assert first.text == "[00:01:00] line 06\n"
    second = store.page(tid, start_s=60.0, offset=first.next_offset, limit=20, timestamps=True)
    assert second.text == "[00:01:10] line 07\n"


@pytest.mark.parametrize(
    "kwargs",
    [{"offset": -1}, {"limit": 0}, {"limit": st.MAX_LIMIT + 1}, {"start_s": 5, "end_s": 1}],
)
def test_bad_paging_arguments(store, kwargs):
    store.finalize(_many(1))
    with pytest.raises(ValueError):
        store.page(f"{KEY}:{VARIANT}", **kwargs)


def test_paging_an_unknown_id_is_a_source_error(store):
    with pytest.raises(SourceError, match="Unknown transcript id"):
        store.page(f"{KEY}:whisper-en-tiny")
