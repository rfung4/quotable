"""Scores for the matrix: WER/CER, timestamp drift, speakers, hallucination flags, pass/fail.

``python -m evals.score`` reads ``<results>/cells.jsonl`` (written by ``evals.run``), compares
every cell's transcript with its reference and writes ``<results>/scores.jsonl``: one row per
cell with the metrics, ``pass`` (``null`` on skipped cells) and ``fails``, the names of the
rules it broke. ``--check`` replays both files and scores nothing.

References: the script for TTS entries (exact text, turn times and speakers), the large-v3
oracle for everything else, and on auto-captioned YouTube entries the captions' own word
timings as a second, independent timing reference. Edit distances are exact Levenshtein on
numpy rows (no ``jiwer``, Decision 5); the alignment path uses Hirschberg's split, so a 3 h
transcript aligns in linear memory.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from evals import oracle
from evals.build_corpus import ROOT, load_manifest
from evals.oracle import EMPTY_BY_CONSTRUCTION, LOOP_OF, MAY_BE_EMPTY

RESULTS = ROOT / "evals" / "results"
REFS = ROOT / "evals" / "refs"

# Decision 10: first values, revisited once after the first full run (changes go in the report).
WER_MAX = {"base": 0.20, "small": 0.15, "turbo": 0.10}  # tiny: reported only
WORD_DRIFT_MAX = 0.2  # TTS turn starts and auto-caption word timings
# Revisited 0.3 → 0.5: the large-v3 oracle starts words early itself, so `small`, which sits
# within 0.1 s of the exact TTS turn starts, measures 0.32–0.36 s against it on TED.
SEGMENT_DRIFT_MAX = 0.5  # first word of each oracle segment
# Added at the revisit: Whisper words over the independent captions reference. The oracle runs
# through the same VAD as the product, so a VAD that drops speech is invisible to WER.
COVERAGE_MIN = 0.8
SPEAKER_MIN = 0.90
REPEAT_EXCESS_MAX = 0.02
LENGTH_RANGE = (0.8, 1.2)
BOUNDARY_SLACK_S = 1.0
FULL_CELLS = 4_000_000  # below this, align with a full matrix and backtrack

_APOSTROPHES = "'’ʼ`"
# Added at the revisit: the fillers Whisper's own EnglishTextNormalizer ignores. `turbo` writes
# them and the `large-v3` oracle mostly does not, which is style, not error (151 of pc20-269's
# 728 insertions). `wer_raw` keeps them.
FILLERS = frozenset({"hmm", "mm", "mhm", "mmm", "uh", "um"})


# --- tokens -----------------------------------------------------------------------------


def normalise(text: str) -> list[str]:
    """Lowercase, punctuation, symbols and fillers out, whitespace collapsed. No number-word
    mapping: ``1960`` and ``nineteen sixty`` stay different, and the report says so."""
    chars = []
    for ch in unicodedata.normalize("NFKC", text).lower():
        if ch in _APOSTROPHES:
            continue
        chars.append(" " if unicodedata.category(ch)[0] in "PSZC" else ch)
    return [tok for tok in "".join(chars).split() if tok not in FILLERS]


@dataclass
class Side:
    """One transcript as tokens, with what is known about each token's time and speaker."""

    raw: list[str] = field(default_factory=list)  # whitespace tokens, case and punctuation kept
    norm: list[str] = field(default_factory=list)
    start: list[float | None] = field(default_factory=list)  # per normalised token
    speaker: list[str | None] = field(default_factory=list)
    seg_first: list[int] = field(default_factory=list)  # first token of each segment
    language: str | None = None


def side(segments: list[dict[str, Any]], language: str | None = None) -> Side:
    s = Side(language=language)
    for seg in segments:
        s.raw += seg["text"].split()
        tokens = normalise(seg["text"])
        if not tokens:
            continue
        timed = [
            (tok, w["start"], w.get("speaker"))
            for w in seg.get("words") or []
            for tok in normalise(w["text"])
        ]
        s.seg_first.append(len(s.norm))
        s.norm += tokens
        if timed and [tok for tok, _, _ in timed] == tokens:
            s.start += [start for _, start, _ in timed]
            s.speaker += [spk or seg.get("speaker") for _, _, spk in timed]
        else:  # no usable word timings: only the segment's first token has a time
            s.start += [seg["start"]] + [None] * (len(tokens) - 1)
            s.speaker += [seg.get("speaker")] * len(tokens)
    return s


def tts_side(reference: dict[str, Any], language: str | None) -> Side:
    turns = [
        {"start": t["start"], "end": t["end"], "text": t["text"], "speaker": t["speaker"]}
        for t in reference["turns"]
    ]
    return side(turns, language)


# --- edit distance ----------------------------------------------------------------------


def _encode(*seqs: list[str]) -> list[np.ndarray]:
    vocab: dict[str, int] = {}
    return [np.array([vocab.setdefault(t, len(vocab)) for t in s], dtype=np.int32) for s in seqs]


def _chars(tokens: list[str]) -> np.ndarray:
    return np.frombuffer(" ".join(tokens).encode("utf-32-le"), dtype=np.uint32)


def _next_row(row: np.ndarray, x: Any, b: np.ndarray, i: int, idx: np.ndarray) -> np.ndarray:
    new = np.empty_like(row)
    new[0] = i
    np.minimum(row[:-1] + (b != x), row[1:] + 1, out=new[1:])
    # Insertions chain along the row: new[j] = min over k <= j of new[k] + (j - k).
    return np.minimum.accumulate(new - idx) + idx


def _last_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Edit distance from all of ``a`` to every prefix of ``b``."""
    idx = np.arange(len(b) + 1, dtype=np.int32)
    row = idx.copy()
    for i, x in enumerate(a, 1):
        row = _next_row(row, x, b, i, idx)
    return row


def distance(a: np.ndarray, b: np.ndarray) -> int:
    if len(a) > len(b):  # loop over the shorter, vectorise over the longer
        a, b = b, a
    return int(_last_row(a, b)[-1])


Pair = tuple[int | None, int | None]  # (ref index, hyp index); None marks a deletion/insertion


def _full(a: np.ndarray, b: np.ndarray) -> list[Pair]:
    idx = np.arange(len(b) + 1, dtype=np.int32)
    d = np.empty((len(a) + 1, len(b) + 1), dtype=np.int32)
    d[0] = idx
    for i, x in enumerate(a, 1):
        d[i] = _next_row(d[i - 1], x, b, i, idx)
    i, j, path = len(a), len(b), []
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (a[i - 1] != b[j - 1]):
            i, j = i - 1, j - 1
            path.append((i, j))
        elif i and d[i, j] == d[i - 1, j] + 1:
            i -= 1
            path.append((i, None))
        else:
            j -= 1
            path.append((None, j))
    path.reverse()
    return path


def _hirschberg(a: np.ndarray, b: np.ndarray, i0: int, j0: int, out: list[Pair]) -> None:
    n, m = len(a), len(b)
    if n == 0 or m == 0 or n == 1 or (n + 1) * (m + 1) <= FULL_CELLS:
        out.extend(
            (None if i is None else i0 + i, None if j is None else j0 + j) for i, j in _full(a, b)
        )
        return
    mid = n // 2
    forward = _last_row(a[:mid], b)
    backward = _last_row(a[mid:][::-1], b[::-1])[::-1]
    k = int(np.argmin(forward + backward))
    _hirschberg(a[:mid], b[:k], i0, j0, out)
    _hirschberg(a[mid:], b[k:], i0 + mid, j0 + k, out)


def align(a: np.ndarray, b: np.ndarray) -> list[Pair]:
    """An optimal edit path in O(len(a) + len(b)) memory."""
    out: list[Pair] = []
    _hirschberg(a, b, 0, 0, out)
    return out


# --- metrics ----------------------------------------------------------------------------


def _ratio(num: float, den: float, digits: int = 4) -> float | None:
    return round(num / den, digits) if den else None


def drift(
    pairs: list[Pair], ref: Side, hyp: Side, points: list[int], period: float | None = None
) -> dict[str, Any]:
    """|Δstart| over reference tokens in ``points`` matched to a hypothesis token, both timed;
    ``bias_s`` is the signed median (negative: the hypothesis is early).

    On a looped reference the alignment may pair a word with the same word one repeat away (every
    such path costs the same), so with ``period`` each Δ is folded into ±period/2 and the folded
    pairs are counted as ``shifted``."""
    wanted = set(points)
    deltas = [
        hyp.start[j] - ref.start[i]
        for i, j in pairs
        if i in wanted
        and j is not None
        and ref.norm[i] == hyp.norm[j]
        and ref.start[i] is not None
        and hyp.start[j] is not None
    ]
    out: dict[str, Any] = {}
    if period:
        folded = [(d + period / 2) % period - period / 2 for d in deltas]
        out["shifted"] = sum(1 for d in deltas if abs(d) > period / 2)
        deltas = folded
    if not deltas:
        return {"median_s": None, "p95_s": None, "bias_s": None, "pairs": 0, **out}
    gaps = np.abs(deltas)
    return {
        "median_s": round(float(np.median(gaps)), 3),
        "p95_s": round(float(np.percentile(gaps, 95)), 3),
        "bias_s": round(float(np.median(deltas)), 3),
        "pairs": len(deltas),
        **out,
    }


def speaker_accuracy(pairs: list[Pair], ref: Side, hyp: Side) -> float | None:
    """Per-word agreement over aligned words under the best ref→hyp label mapping."""
    labelled = [
        (ref.speaker[i], hyp.speaker[j]) for i, j in pairs if i is not None and j is not None
    ]
    if not labelled:
        return None
    counts = Counter(labelled)
    ref_labels = sorted({r for r, _ in labelled if r is not None})
    hyp_labels = sorted({h for _, h in labelled if h is not None})
    best = 0
    slots: list[str | None] = [*hyp_labels, *([None] * len(ref_labels))]
    for chosen in itertools.permutations(slots, len(ref_labels)):
        score = sum(
            counts[(r, h)] for r, h in zip(ref_labels, chosen, strict=True) if h is not None
        )
        best = max(best, score)
    return round(best / len(labelled), 4)


def turns_missed(ref: Side, hyp: Side) -> tuple[int, int]:
    """(reference speaker changes with no hypothesis change within 1 s, reference changes)."""
    boundaries = [
        ref.start[k]
        for prev, k in itertools.pairwise(ref.seg_first)
        if ref.speaker[k] != ref.speaker[prev] and ref.start[k] is not None
    ]
    changes, last = [], None
    for spk, start in zip(hyp.speaker, hyp.start, strict=True):
        if spk is None or start is None:
            continue
        if last is not None and spk != last:
            changes.append(start)
        last = spk
    missed = sum(1 for b in boundaries if not any(abs(c - b) <= BOUNDARY_SLACK_S for c in changes))
    return missed, len(boundaries)


def repeat_rate(tokens: list[str], n: int = 4, min_count: int = 3) -> float:
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    counts = Counter(grams)
    return sum(1 for g in grams if counts[g] >= min_count) / len(grams)


def metrics(
    ref: Side,
    hyp: Side,
    *,
    ref_kind: str,
    asr: Side | None = None,
    captions: Side | None = None,
    period: float | None = None,
    speakers: bool = False,
) -> dict[str, Any]:
    """Every number for one cell. ``ref_kind`` is ``tts`` (exact turn starts, judged at word
    level) or ``oracle`` (Whisper's own timings, judged at segment level). ``captions`` is the
    entry's captions or transcript-tag reference, ``asr`` the same when it carries word timings."""
    a, b = _encode(ref.norm, hyp.norm)
    pairs = align(a, b)
    subs = sum(1 for i, j in pairs if i is not None and j is not None and a[i] != b[j])
    dels = sum(1 for _, j in pairs if j is None)
    ins = sum(1 for i, _ in pairs if i is None)
    out: dict[str, Any] = {
        "ref_tokens": len(ref.norm),
        "hyp_tokens": len(hyp.norm),
        "subs": subs,
        "dels": dels,
        "ins": ins,
        "wer": _ratio(subs + dels + ins, len(ref.norm)),
        "wer_raw": _ratio(distance(*_encode(ref.raw, hyp.raw)), len(ref.raw)),
        "cer": _ratio(distance(_chars(ref.norm), _chars(hyp.norm)), len(" ".join(ref.norm))),
        "length_ratio": _ratio(len(hyp.norm), len(ref.norm), 3),
        "repeat_excess": round(repeat_rate(hyp.norm) - repeat_rate(ref.norm), 4),
        "language": hyp.language,
    }
    found = drift(pairs, ref, hyp, ref.seg_first, period)
    out["drift_level"] = "word" if ref_kind == "tts" else "segment"
    out.update({f"drift_{k}": v for k, v in found.items()})
    if captions is not None:
        out["captions_coverage"] = _ratio(len(hyp.norm), len(captions.norm), 3)
    if asr is not None:
        asr_pairs = align(*_encode(asr.norm, hyp.norm))
        found = drift(asr_pairs, asr, hyp, list(range(len(asr.norm))))
        out.update({f"asr_drift_{k}": v for k, v in found.items()})
    if speakers:
        out["speaker_accuracy"] = speaker_accuracy(pairs, ref, hyp)
        out["turns_missed"], out["ref_turn_changes"] = turns_missed(ref, hyp)
        out["hyp_speakers"] = len({s for s in hyp.speaker if s is not None})
    return out


# --- judging ----------------------------------------------------------------------------


def _same_language(found: str | None, expected: str) -> bool | None:
    if not found or found == "und":
        return None  # a feed transcript names no language; nothing to judge
    return found.split("-")[0].lower() == expected.split("-")[0].lower()


def judge(
    row: dict[str, Any], entry: dict[str, Any], m: dict[str, Any]
) -> tuple[bool | None, list[str]]:
    """``(pass, fails)``. Skipped cells are ``None``; crash, fail and error cells fail, except
    that a no-speech error is the right answer on silence and on the 0.6 s clip, and a
    no-captions error is the right answer where the manifest says the default track is
    refused (several caption languages, none named as the video's)."""
    status, eid = row["status"], entry["id"]
    if status == "skipped":
        return None, []
    error_type = (row.get("error") or {}).get("type") if status == "error" else None
    if entry["expect"].get("default_captions") == "refused" and row["mode"] == "captions":
        return (True, []) if error_type == "NoCaptionsError" else (False, ["status"])
    no_speech = error_type == "TranscriptionError"
    if eid in EMPTY_BY_CONSTRUCTION:
        if no_speech or (status == "ok" and m.get("hyp_tokens") == 0):
            return True, []
        return False, ["silence" if status == "ok" else "status"]
    if status != "ok":
        accepted = no_speech and eid in MAY_BE_EMPTY
        return accepted, [] if accepted else ["status"]
    if eid in MAY_BE_EMPTY:  # shorter than a VAD window: it must not crash, nothing more
        return True, []
    if row["model"] == "tiny":
        return True, []  # reported only

    fails = []
    if _same_language(m.get("language"), entry["expect"]["language"]) is False:
        fails.append("language")
    if row["mode"] == "captions":  # agreement with the oracle is reported, not judged
        return not fails, fails
    lo, hi = LENGTH_RANGE
    if m["length_ratio"] is None or not lo <= m["length_ratio"] <= hi:
        fails.append("length")
    if m["repeat_excess"] > REPEAT_EXCESS_MAX:
        fails.append("repeats")
    if row["real_path"]:
        # The live download proves the path; a feed can splice in other ads than the
        # cached copy had, so text and timing are judged on the cached-audio cells.
        return not fails, fails
    if m["wer"] is None or m["wer"] > WER_MAX[row["model"]]:
        fails.append("wer")
    if m.get("captions_coverage") is not None and m["captions_coverage"] < COVERAGE_MIN:
        fails.append("coverage")  # one-sided: a feed's audio may carry ads its transcript lacks
    limit = WORD_DRIFT_MAX if m["drift_level"] == "word" else SEGMENT_DRIFT_MAX
    if m["drift_median_s"] is not None and m["drift_median_s"] > limit:
        fails.append("drift")
    if m.get("asr_drift_median_s") is not None and m["asr_drift_median_s"] > WORD_DRIFT_MAX:
        fails.append("asr_drift")
    if m.get("speaker_accuracy") is not None and m["speaker_accuracy"] < SPEAKER_MIN:
        fails.append("speakers")
    return not fails, fails


# --- files ------------------------------------------------------------------------------


def digest(row: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(row, sort_keys=True).encode()).hexdigest()[:16]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def reference(entry: dict[str, Any]) -> tuple[str, Side, float | None]:
    """``(kind, side, period)``; ``period`` is the repeat length of a looped reference."""
    data = _read_json(REFS / f"{entry['id']}.json")
    language = entry["expect"]["language"]
    if entry["kind"] == "tts":
        return "tts", tts_side(data, language), None
    period = None
    if entry["id"] in LOOP_OF:
        once = len(_read_json(REFS / f"{LOOP_OF[entry['id']]}.json")["segments"])
        period = data["segments"][once]["start"] - data["segments"][0]["start"]
    return "oracle", side(data["segments"], data.get("language")), period


def captions_reference(entry: dict[str, Any]) -> Side | None:
    """The captions or transcript-tag reference Task 3 stored, when the entry has one."""
    if not oracle.wants_captions_ref(entry):
        return None
    data = _read_json(REFS / f"{entry['id']}.captions.json")
    return side(data["segments"], data.get("language"))


def score_row(row: dict[str, Any], entry: dict[str, Any], results: Path) -> dict[str, Any]:
    out = {k: row[k] for k in ("cell_id", "entry", "mode", "model", "device", "real_path")}
    out.update(cell_digest=digest(row), status=row["status"])
    m: dict[str, Any] = {}
    if row["status"] == "ok":
        kind, ref, period = reference(entry)
        data = _read_json(results / row["transcript_path"])
        hyp = side(data["segments"], data.get("language"))
        captions = captions_reference(entry) if row["mode"] == "whisper" else None
        asr = captions if entry["expect"]["captions"] == "asr" else None
        m = {
            "reference": kind,
            **metrics(
                ref,
                hyp,
                ref_kind=kind,
                asr=asr,
                captions=captions,
                period=period,
                speakers=row["diarize"] and kind == "tts",
            ),
        }
    out.update(m)
    out["pass"], out["fails"] = judge(row, entry, m)
    return out


def _missing_references(rows: list[dict[str, Any]], entries: dict[str, Any]) -> list[str]:
    missing = set()
    for row in rows:
        entry = entries[row["entry"]]
        if row["status"] != "ok":
            continue
        if not (REFS / f"{entry['id']}.json").is_file():
            missing.add(f"{entry['id']}.json")
        if oracle.wants_captions_ref(entry) and row["mode"] == "whisper":
            if not (REFS / f"{entry['id']}.captions.json").is_file():
                missing.add(f"{entry['id']}.captions.json")
    return sorted(missing)


def write_scores(results: Path) -> int:
    rows = load_jsonl(results / "cells.jsonl")
    if not rows:
        print(f"evals.score: no rows in {results / 'cells.jsonl'}", file=sys.stderr)
        return 1
    entries = {e["id"]: e for e in load_manifest()}
    missing = _missing_references(rows, entries)
    if missing:
        print("evals.score: references missing:", *missing, sep="\n  ", file=sys.stderr)
        return 1
    scores = []
    for row in rows:
        scores.append(score_row(row, entries[row["entry"]], results))
        mark = {True: "pass", False: "FAIL", None: "skip"}[scores[-1]["pass"]]
        fails = ",".join(scores[-1]["fails"])
        print(f"{mark}  {row['cell_id']:<52} wer {scores[-1].get('wer')}  {fails}", flush=True)
    path = results / "scores.jsonl"
    partial = path.with_name("partial-" + path.name)
    partial.write_text("".join(json.dumps(s) + "\n" for s in scores), encoding="utf-8")
    os.replace(partial, path)
    return _report(scores)


def _report(scores: list[dict[str, Any]]) -> int:
    counts = Counter({True: "pass", False: "fail", None: "skipped"}[s["pass"]] for s in scores)
    print("  ".join(f"{k} {counts.get(k, 0)}" for k in ("pass", "fail", "skipped")))
    return 0


def check(results: Path) -> int:
    rows = load_jsonl(results / "cells.jsonl")
    if not rows:
        print(f"FAIL  no rows in {results / 'cells.jsonl'}")
        return 1
    scores = {s["cell_id"]: s for s in load_jsonl(results / "scores.jsonl")}
    problems = []
    for row in rows:
        cid, scored = row["cell_id"], scores.get(row["cell_id"])
        if scored is None:
            problems.append(f"{cid}: not scored")
        elif scored["cell_digest"] != digest(row):
            problems.append(f"{cid}: scored against an older row (re-run evals.score)")
        if row["status"] == "crash":
            problems.append(f"{cid}: crash ({row['reason']})")
        if row["status"] == "skipped" and row["reason"] != "no token":
            problems.append(f"{cid}: skipped for {row['reason']!r}; only a missing token may skip")
        if row["real_path"] and scored is not None and scored["pass"] is not True:
            problems.append(f"{cid}: real-path cell did not pass ({', '.join(scored['fails'])})")
    for p in problems:
        print(f"FAIL  {p}")
    counts = Counter(
        {True: "pass", False: "fail", None: "skipped"}[scores[r["cell_id"]]["pass"]]
        for r in rows
        if r["cell_id"] in scores
    )
    print("  ".join(f"{k} {counts.get(k, 0)}" for k in ("pass", "fail", "skipped")))
    print(f"{len(rows)} cells, {len(problems)} problems")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals.score", description=__doc__)
    parser.add_argument("--results", default=str(RESULTS), help="directory with cells.jsonl")
    parser.add_argument("--check", action="store_true", help="replay the files; score nothing")
    args = parser.parse_args(argv)
    results = Path(args.results).resolve()
    return check(results) if args.check else write_scores(results)


if __name__ == "__main__":
    raise SystemExit(main())
