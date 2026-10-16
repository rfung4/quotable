"""The eval scorer on synthetic inputs: distances, alignment, drift, speakers, rules, --check."""

import json
import random

import pytest

from evals import score as sc


def _seg(start, text, words=None, speaker=None):
    seg = {"start": start, "end": start + 1.0, "text": text, "speaker": speaker}
    if words is not None:
        seg["words"] = [
            {"start": s, "end": s + 0.2, "text": w, "speaker": speaker} for s, w in words
        ]
    return seg


def _timed(start, text, step=0.3, speaker=None):
    words = [(start + k * step, w) for k, w in enumerate(text.split())]
    return _seg(start, text, words, speaker)


def _lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


# --- tokens and distances ---------------------------------------------------------------


def test_normalise_drops_case_punctuation_and_symbols():
    assert sc.normalise("Hello, World! It's SHA-256 — $48,500.") == [
        "hello",
        "world",
        "its",
        "sha",
        "256",
        "48",
        "500",
    ]
    assert sc.normalise("  ¿Qué   tal?\n") == ["qué", "tal"]
    assert sc.normalise("So, uh, I think, um... Hmm. Umbrella") == ["so", "i", "think", "umbrella"]


@pytest.mark.parametrize(
    ("a", "b", "want"),
    [
        ("kitten", "sitting", 3),
        ("", "abc", 3),
        ("abc", "", 3),
        ("flaw", "lawn", 2),
        ("same", "same", 0),
    ],
)
def test_known_edit_distances(a, b, want):
    assert sc.distance(*sc._encode(list(a), list(b))) == want


def test_alignment_is_optimal_through_deep_recursion(monkeypatch):
    monkeypatch.setattr(sc, "FULL_CELLS", 30)  # small inputs still take the Hirschberg split
    rng = random.Random(3)
    for _ in range(200):
        a = [rng.choice("abc") for _ in range(rng.randint(0, 20))]
        b = [rng.choice("abc") for _ in range(rng.randint(0, 20))]
        pairs = sc.align(*sc._encode(a, b))
        cost = sum(1 for i, j in pairs if i is None or j is None or a[i] != b[j])
        assert cost == _lev(a, b)
        assert [i for i, _ in pairs if i is not None] == list(range(len(a)))
        assert [j for _, j in pairs if j is not None] == list(range(len(b)))


def test_wer_counts_and_raw_versus_normalised():
    ref = sc.side([_seg(0.0, "The cat sat on the mat.")])
    hyp = sc.side([_seg(0.0, "the cat sat on a mat")])
    m = sc.metrics(ref, hyp, ref_kind="oracle")
    assert (m["subs"], m["dels"], m["ins"]) == (1, 0, 0)
    assert m["wer"] == round(1 / 6, 4)
    assert m["wer_raw"] == round(3 / 6, 4)  # "The", "the"→"a" and "mat." all differ raw
    assert m["length_ratio"] == 1.0

    shorter = sc.metrics(ref, sc.side([_seg(0.0, "the cat sat")]), ref_kind="oracle")
    assert (shorter["dels"], shorter["wer"], shorter["length_ratio"]) == (3, 0.5, 0.5)


# --- drift --------------------------------------------------------------------------------


def test_drift_of_shifted_segments_at_segment_and_word_level():
    texts = ["one two three", "four five six", "seven eight nine"]
    ref = sc.side([_timed(10.0 * k, t) for k, t in enumerate(texts)])
    hyp = sc.side([_timed(10.0 * k + 0.5, t) for k, t in enumerate(texts)])

    oracle = sc.metrics(ref, hyp, ref_kind="oracle")
    assert oracle["drift_level"] == "segment"
    assert (oracle["drift_median_s"], oracle["drift_pairs"]) == (0.5, 3)  # one per segment

    tts = sc.metrics(sc.side([_seg(10.0 * k, t) for k, t in enumerate(texts)]), hyp, ref_kind="tts")
    assert tts["drift_level"] == "word" and tts["drift_median_s"] == 0.5

    asr = sc.metrics(ref, hyp, ref_kind="oracle", asr=ref)
    assert (asr["asr_drift_median_s"], asr["asr_drift_pairs"]) == (0.5, 9)  # every word


def test_drift_skips_untimed_hypothesis_words():
    ref = sc.side([_timed(0.0, "alpha beta"), _timed(5.0, "gamma delta")])
    hyp = sc.side([_seg(0.2, "alpha beta gamma delta")])  # manual captions: one start only
    m = sc.metrics(ref, hyp, ref_kind="oracle")
    assert (m["drift_pairs"], m["drift_median_s"]) == (1, 0.2)


def test_drift_reports_the_signed_bias():
    texts = ["one two", "three four", "five six"]
    ref = sc.side([_timed(10.0 * k, t) for k, t in enumerate(texts)])
    early = sc.side([_timed(10.0 * k - 0.3, t) for k, t in enumerate(texts)])
    m = sc.metrics(ref, early, ref_kind="oracle")
    assert (m["drift_median_s"], m["drift_bias_s"]) == (0.3, -0.3)
    assert "drift_shifted" not in m


def test_drift_on_a_loop_folds_pairs_one_repeat_away():
    ref = sc.side([_timed(11.0 * k, "and so my fellow americans") for k in range(4)])
    hyp = sc.side([_timed(11.0 * k + 0.2, "and so my fellow americans") for k in range(4)])
    # The path a periodic alignment may take: every word paired with the next repeat's word.
    pairs = [(i, i + 5 if i + 5 < 20 else None) for i in range(20)]
    unfolded = sc.drift(pairs, ref, hyp, ref.seg_first)
    assert unfolded["median_s"] == 11.2
    folded = sc.drift(pairs, ref, hyp, ref.seg_first, period=11.0)
    assert (folded["median_s"], folded["bias_s"], folded["shifted"], folded["pairs"]) == (
        0.2,
        0.2,
        3,
        3,
    )


def test_captions_coverage_is_hypothesis_words_over_the_captions_reference():
    ref = sc.side([_seg(0.0, "a b c d")])  # the oracle shares the product's VAD
    captions = sc.side([_seg(0.0, "a b c d e f g h i j k l m n o p")])
    m = sc.metrics(ref, ref, ref_kind="oracle", captions=captions)
    assert (m["wer"], m["captions_coverage"]) == (0.0, 0.25)
    assert "captions_coverage" not in sc.metrics(ref, ref, ref_kind="oracle")


# --- speakers -----------------------------------------------------------------------------


def _speakers(labels):
    return sc.side([_timed(float(k), f"w{k}", speaker=lab) for k, lab in enumerate(labels)])


def test_speaker_accuracy_ignores_label_names():
    ref = _speakers(["dana", "dana", "marcus", "marcus"])
    for hyp_labels in (["S0", "S0", "S1", "S1"], ["S1", "S1", "S0", "S0"]):
        m = sc.metrics(ref, _speakers(hyp_labels), ref_kind="tts", speakers=True)
        assert m["speaker_accuracy"] == 1.0
        assert m["hyp_speakers"] == 2


def test_speaker_accuracy_counts_wrong_and_extra_labels():
    ref = _speakers(["a", "a", "a", "b", "b", "b"])
    m = sc.metrics(ref, _speakers(["X", "X", "Y", "Y", "Y", "Z"]), ref_kind="tts", speakers=True)
    assert m["speaker_accuracy"] == round(4 / 6, 4)  # best map a→X, b→Y
    one_speaker = sc.metrics(
        _speakers(["s"] * 4), _speakers(["P", "P", "P", "Q"]), ref_kind="tts", speakers=True
    )
    assert one_speaker["speaker_accuracy"] == 0.75  # over-splitting one voice costs words


def test_turn_changes_missed_by_more_than_a_second():
    ref = sc.side([_seg(0.0, "hello there", speaker="a"), _seg(5.0, "hi back", speaker="b")])
    near = sc.side([_timed(0.0, "hello there", speaker="S0"), _timed(5.6, "hi back", speaker="S1")])
    far = sc.side([_timed(0.0, "hello there", speaker="S0"), _timed(6.5, "hi back", speaker="S1")])
    assert sc.turns_missed(ref, near) == (0, 1)
    assert sc.turns_missed(ref, far) == (1, 1)


# --- hallucination flags ------------------------------------------------------------------


def test_repeat_detection_is_relative_to_the_reference():
    ref = sc.side([_seg(0.0, "we choose to go to the moon in this decade")])
    loop = "and so my fellow americans " * 4
    hyp = sc.side([_seg(0.0, "we choose to go to the moon in this decade " + loop)])
    assert sc.metrics(ref, hyp, ref_kind="oracle")["repeat_excess"] > sc.REPEAT_EXCESS_MAX
    assert sc.metrics(ref, ref, ref_kind="oracle")["repeat_excess"] == 0.0
    looped = sc.side([_seg(0.0, loop)])
    assert sc.metrics(looped, looped, ref_kind="oracle")["repeat_excess"] == 0.0


# --- rules --------------------------------------------------------------------------------


def _row(**kw):
    row = {"status": "ok", "model": "turbo", "mode": "whisper", "real_path": False, "error": None}
    return {**row, **kw}


def _entry(eid="talk", language="en"):
    return {"id": eid, "expect": {"language": language, "captions": "none"}}


def _m(**kw):
    m = {
        "hyp_tokens": 100,
        "language": "en",
        "wer": 0.05,
        "length_ratio": 1.0,
        "repeat_excess": 0.0,
        "drift_level": "segment",
        "drift_median_s": 0.1,
    }
    return {**m, **kw}


def test_refused_default_captions_rule():
    ted = {
        "id": "ted",
        "expect": {"language": "en", "captions": "manual", "default_captions": "refused"},
    }
    refused = {"type": "NoCaptionsError", "message": "names none"}
    captions = {"mode": "captions", "model": None}
    assert sc.judge(_row(status="error", error=refused, **captions), ted, {}) == (True, [])
    # a track came back after all (the bug), or another error: both fail
    assert sc.judge(_row(**captions), ted, _m(language="sq")) == (False, ["status"])
    download = {"type": "DownloadError", "message": "403"}
    assert sc.judge(_row(status="error", error=download, **captions), ted, {}) == (
        False,
        ["status"],
    )
    # Whisper cells on the same entry are judged as usual
    assert sc.judge(_row(), ted, _m()) == (True, [])


def test_silence_rule():
    silence = _entry("silence-30s")
    no_speech = {"type": "TranscriptionError", "message": "No speech", "fix": "x"}
    assert sc.judge(_row(status="error", error=no_speech), silence, {}) == (True, [])
    assert sc.judge(_row(model="tiny"), silence, _m(hyp_tokens=3)) == (False, ["silence"])
    assert sc.judge(_row(status="crash"), silence, {}) == (False, ["status"])
    download = {"type": "DownloadError", "message": "403", "fix": "x"}
    assert sc.judge(_row(status="error", error=download), silence, {}) == (False, ["status"])


@pytest.mark.parametrize(
    ("row", "m", "want"),
    [
        (_row(), _m(), (True, [])),
        (_row(), _m(wer=0.12), (False, ["wer"])),
        (_row(model="base"), _m(wer=0.12), (True, [])),
        (_row(model="tiny"), _m(wer=0.6, language="de", repeat_excess=0.5), (True, [])),
        (_row(real_path=True), _m(wer=0.4, drift_median_s=3.0), (True, [])),
        (_row(real_path=True), _m(length_ratio=0.5), (False, ["length"])),
        (_row(), _m(repeat_excess=0.05), (False, ["repeats"])),
        (_row(), _m(drift_median_s=0.45), (True, [])),
        (_row(), _m(drift_median_s=0.55), (False, ["drift"])),
        (_row(), _m(drift_level="word", drift_median_s=0.25), (False, ["drift"])),
        (_row(), _m(asr_drift_median_s=0.25), (False, ["asr_drift"])),
        (_row(), _m(speaker_accuracy=0.8), (False, ["speakers"])),
        (_row(), _m(captions_coverage=0.24), (False, ["coverage"])),
        (_row(), _m(captions_coverage=1.4), (True, [])),  # a feed's audio carries extra ads
        (_row(real_path=True), _m(captions_coverage=0.24), (True, [])),
        (_row(model="tiny"), _m(captions_coverage=0.24), (True, [])),
        (_row(mode="captions", model=None), _m(language="sq", wer=0.9), (False, ["language"])),
        (_row(mode="captions", model=None), _m(language=None, wer=0.9), (True, [])),
        (_row(status="skipped"), {}, (None, [])),
        (_row(status="crash"), {}, (False, ["status"])),
        (_row(status="fail"), {}, (False, ["status"])),
        (_row(status="error", error={"type": "DownloadError"}), {}, (False, ["status"])),
    ],
)
def test_threshold_application(row, m, want):
    assert sc.judge(row, _entry(), m) == want


def test_a_clip_shorter_than_a_vad_window_only_has_to_not_crash():
    clip = _entry("jfk-clip-0.6s")
    no_speech = {"type": "TranscriptionError", "message": "No speech", "fix": "x"}
    assert sc.judge(_row(status="error", error=no_speech), clip, {}) == (True, [])
    assert sc.judge(_row(), clip, _m(wer=1.0, language="cy")) == (True, [])
    assert sc.judge(_row(status="crash"), clip, {}) == (False, ["status"])


# --- --check ------------------------------------------------------------------------------


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _cell(cell_id, **kw):
    return {"cell_id": cell_id, "status": "ok", "reason": None, "real_path": False, **kw}


def test_check_fails_without_rows(tmp_path):
    assert sc.check(tmp_path) == 1
    (tmp_path / "cells.jsonl").write_text("", encoding="utf-8")
    assert sc.check(tmp_path) == 1


def test_check_requires_current_scores_and_passing_real_path_cells(tmp_path):
    cells = [
        _cell("a|whisper|turbo|cuda|false|false"),
        _cell("a|whisper|turbo|cuda|false|true", real_path=True),
        _cell("a|whisper|turbo|cuda|true|false", status="skipped", reason="no token"),
    ]
    _write(tmp_path / "cells.jsonl", cells)
    verdicts = [False, True, None]  # an ordinary cell may fail; that is a finding, not a gate
    scores = [
        {"cell_id": c["cell_id"], "cell_digest": sc.digest(c), "pass": p, "fails": []}
        for c, p in zip(cells, verdicts, strict=True)
    ]
    _write(tmp_path / "scores.jsonl", scores)
    assert sc.check(tmp_path) == 0

    _write(tmp_path / "scores.jsonl", [{**scores[0]}, {**scores[1], "pass": False}, scores[2]])
    assert sc.check(tmp_path) == 1  # the real-path cell must pass

    _write(tmp_path / "scores.jsonl", scores[:2])
    assert sc.check(tmp_path) == 1  # a cell without a score

    _write(tmp_path / "cells.jsonl", [{**cells[0], "wall_s": 9.9}, *cells[1:]])
    _write(tmp_path / "scores.jsonl", scores)
    assert sc.check(tmp_path) == 1  # the row changed after scoring

    _write(tmp_path / "cells.jsonl", [cells[0], cells[1], {**cells[2], "reason": "no media"}])
    assert sc.check(tmp_path) == 1  # only a missing token may skip

    crashed = {**cells[0], "status": "crash", "reason": "exit 1"}
    _write(tmp_path / "cells.jsonl", [crashed, *cells[1:]])
    _write(
        tmp_path / "scores.jsonl", [{**scores[0], "cell_digest": sc.digest(crashed)}, *scores[1:]]
    )
    assert sc.check(tmp_path) == 1
