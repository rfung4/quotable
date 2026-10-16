import sys
import types
from pathlib import Path

import pytest

from quotable import diarize as dz
from quotable.config import Settings
from quotable.diarize import Turn, assign_speakers
from quotable.models import DiarizationUnavailableError, Segment, Word


def _seg(*words: tuple[float, float, str]) -> Segment:
    ws = [Word(start=a, end=b, text=t) for a, b, t in words]
    return Segment(start=ws[0].start, end=ws[-1].end, text=" ".join(w.text for w in ws), words=ws)


# --- assignment -----------------------------------------------------------------------


def test_each_word_gets_the_turn_it_overlaps_most():
    turns = [Turn(0.0, 2.0, "A"), Turn(2.0, 5.0, "B")]
    (seg,) = assign_speakers(
        [_seg((0.5, 1.0, "hi"), (1.8, 2.6, "there"), (3.0, 3.5, "you"))], turns
    )
    assert [w.speaker for w in seg.words] == ["A", "B", "B"]  # "there": 0.2 s in A, 0.6 s in B


def test_equal_overlap_goes_to_the_earlier_turn():
    turns = [Turn(2.0, 5.0, "B"), Turn(0.0, 2.0, "A")]  # unsorted on purpose
    (seg,) = assign_speakers([_seg((1.5, 2.5, "split"))], turns)
    assert seg.words[0].speaker == "A"


def test_gap_words_take_the_nearest_turn_within_one_second():
    turns = [Turn(0.0, 1.0, "A"), Turn(5.0, 6.0, "B")]
    (seg,) = assign_speakers(
        [_seg((1.4, 1.6, "near-a"), (4.0, 4.2, "near-b"), (2.5, 2.7, "lost"))], turns
    )
    assert [w.speaker for w in seg.words] == ["A", "B", None]


def test_exactly_one_second_away_still_counts():
    (seg,) = assign_speakers([_seg((2.0, 2.0, "x"))], [Turn(0.0, 1.0, "A")])
    assert seg.words[0].speaker == "A"


def test_zero_length_word_inside_a_turn_is_labelled():
    (seg,) = assign_speakers([_seg((0.5, 0.5, "um"))], [Turn(0.0, 1.0, "A")])
    assert seg.words[0].speaker == "A"


def test_segment_speaker_is_the_word_majority_ties_to_first_seen():
    turns = [Turn(0.0, 1.0, "A"), Turn(1.0, 3.0, "B")]
    majority, tie = assign_speakers(
        [
            _seg((0.1, 0.5, "a"), (1.1, 1.5, "b"), (1.6, 2.0, "c")),
            _seg((0.6, 0.9, "a"), (1.1, 1.4, "b")),
        ],
        turns,
    )
    assert majority.speaker == "B"
    assert tie.speaker == "A"


def test_wordless_segment_uses_its_own_span_and_none_when_no_turns():
    seg = Segment(start=0.0, end=1.0, text="caption line")
    assert assign_speakers([seg], [Turn(0.2, 0.9, "A")])[0].speaker == "A"
    assert assign_speakers([seg], [])[0].speaker is None


def test_input_segments_are_not_mutated():
    seg = _seg((0.1, 0.2, "x"))
    assign_speakers([seg], [Turn(0.0, 1.0, "A")])
    assert seg.speaker is None and seg.words[0].speaker is None


def test_long_overlapping_turn_found_by_the_index():
    # A turn that starts long before the word must still be considered.
    turns = [Turn(0.0, 100.0, "A")] + [Turn(10.0 + i, 10.5 + i, "B") for i in range(50)]
    (seg,) = assign_speakers([_seg((80.6, 80.9, "late"))], turns)
    assert seg.words[0].speaker == "A"


# --- pipeline loading -----------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_pipeline(monkeypatch):
    monkeypatch.setattr(dz, "_pipeline", None)


def _fake_pyannote(monkeypatch, from_pretrained):
    pkg = types.ModuleType("pyannote")
    mod = types.ModuleType("pyannote.audio")
    mod.Pipeline = types.SimpleNamespace(from_pretrained=from_pretrained)
    pkg.audio = mod
    monkeypatch.setitem(sys.modules, "pyannote", pkg)
    monkeypatch.setitem(sys.modules, "pyannote.audio", mod)


def test_missing_extra_says_how_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyannote.audio", None)  # makes the import fail
    with pytest.raises(DiarizationUnavailableError) as exc:
        dz.load_pipeline("hf_x")
    assert "quotable[diarize]" in exc.value.fix


def test_missing_token_points_at_the_terms_page(monkeypatch):
    _fake_pyannote(monkeypatch, lambda *a, **k: pytest.fail("must not call the Hub"))
    with pytest.raises(DiarizationUnavailableError) as exc:
        dz.load_pipeline(None)
    assert dz.TERMS_URL in exc.value.fix and "QUOTABLE_HF_TOKEN" in exc.value.fix


def test_gated_repo_returning_none_is_an_actionable_error(monkeypatch):
    _fake_pyannote(monkeypatch, lambda *a, **k: None)
    with pytest.raises(DiarizationUnavailableError) as exc:
        dz.load_pipeline("hf_not_accepted")
    assert dz.TERMS_URL in exc.value.fix


def test_401_from_the_hub_is_an_actionable_error(monkeypatch):
    def refuse(*a, **k):
        raise OSError("401 Client Error: Unauthorized for url")

    _fake_pyannote(monkeypatch, refuse)
    with pytest.raises(DiarizationUnavailableError) as exc:
        dz.load_pipeline("hf_bad")
    assert dz.TERMS_URL in exc.value.fix


def test_pipeline_is_loaded_with_the_token_and_cached(monkeypatch):
    calls = []

    class FakePipeline:
        def to(self, device):
            pass

    def from_pretrained(model_id, token=None):
        calls.append((model_id, token))
        return FakePipeline()

    _fake_pyannote(monkeypatch, from_pretrained)
    settings = _settings(hf_token="hf_ok")
    assert dz.get_pipeline(settings) is dz.get_pipeline(settings)
    assert calls == [(dz.MODEL_ID, "hf_ok")]


def _settings(**over):
    base = dict(output_dir=Path("out"), model="tiny", device="cpu", compute_type="int8")
    return Settings(**{**base, **over})


@pytest.mark.parametrize(
    ("device", "cuda_broken", "moved"),
    [("cuda", False, True), ("cpu", False, False), ("cuda", True, False)],
)
def test_pipeline_goes_to_cuda_only_when_whisper_is_on_it(monkeypatch, device, cuda_broken, moved):
    moves = []

    class FakePipeline:
        def to(self, dev):
            moves.append(str(dev))

    _fake_pyannote(monkeypatch, lambda model_id, token=None: FakePipeline())
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True), device=lambda name: name
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(dz.tx, "_cuda_broken", cuda_broken)
    dz.get_pipeline(_settings(hf_token="hf_ok", device=device))
    assert moves == (["cuda"] if moved else [])


def test_diarize_reads_the_exclusive_annotation_sorted(monkeypatch, tmp_path):
    class Span:
        def __init__(self, s, e):
            self.start, self.end = s, e

    class Annotation:
        def __init__(self, tracks):
            self.tracks = tracks

        def itertracks(self, yield_label=False):
            return iter(self.tracks)

    output = types.SimpleNamespace(
        exclusive_speaker_diarization=Annotation(
            [(Span(2.0, 3.0), "t1", "SPEAKER_01"), (Span(0.0, 2.0), "t0", "SPEAKER_00")]
        ),
        speaker_diarization=Annotation([]),
    )
    monkeypatch.setattr(dz, "get_pipeline", lambda settings: lambda audio: output)
    monkeypatch.setattr(dz, "_waveform", lambda path: {"waveform": None, "sample_rate": 16000})
    settings = types.SimpleNamespace(hf_token="hf_ok")
    assert dz.diarize(tmp_path / "a.flac", settings) == [
        Turn(0.0, 2.0, "SPEAKER_00"),
        Turn(2.0, 3.0, "SPEAKER_01"),
    ]
