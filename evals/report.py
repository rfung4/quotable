"""Report tables for the matrix, the scenarios and the MCP probe, and the README benchmark rows.

``python -m evals.report`` reads ``<results>/cells.jsonl``, ``scores.jsonl``,
``scenarios.json``, ``mcp_surface.json`` and ``mcp_checklist.md`` with ``evals/corpus.json``
and writes ``<results>/report.md``: the hardware line, the corpus with licences, a summary
per model and device, one results table per input kind, captions against the oracle,
diarization, the scenarios and the MCP checklist. Where ``docs/report/quotable-eval.md``
carries a ``<!-- MERGE_START(TABLES) -->`` block, that block is replaced by the same tables;
the judgment around it is written by hand. ``--readme`` rewrites the README's Benchmark
section from the manifest's ``readme_row`` entries, so the README's numbers come from the
matrix and name the item they measured. ``--check``
replays: the report names every cell and its tables are current, the README has no pending
benchmark row, and nothing tracked outside the dev docs mentions the deleted benchmark script.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from evals.build_corpus import ROOT, load_manifest
from evals.run import MODELS, RESULTS
from evals.score import load_jsonl

DOC = ROOT / "docs" / "report" / "quotable-eval.md"
README = ROOT / "README.md"
TABLES_START, TABLES_END = "<!-- MERGE_START(TABLES) -->", "<!-- MERGE_END(TABLES) -->"
README_SECTION = "## Benchmark"
KINDS = (
    ("file", "Local files"),
    ("tts", "TTS dialogues"),
    ("youtube", "YouTube"),
    ("feed", "Feeds"),
)
# Cells the per-model summary leaves out: short clips (load dominates, and the eleven JFK
# variants are one 11 s recording) and the synthetic 3 h repeat.
SUMMARY_MIN_AUDIO_S = 200
SUMMARY_EXCLUDE = frozenset({"jfk-loop-3h"})
README_WHISPER = (("base", "cpu"), ("turbo", "cuda"))
PASS = {True: "pass", False: "FAIL", None: "—"}


# --- loading ----------------------------------------------------------------------------


class Data:
    def __init__(self, results: Path, entries: list[dict[str, Any]]):
        self.results = results
        self.entries = {e["id"]: e for e in entries}
        self.cells = load_jsonl(results / "cells.jsonl")
        self.scores = {s["cell_id"]: s for s in load_jsonl(results / "scores.jsonl")}
        self.scenarios = _read_json(results / "scenarios.json")
        self.surface = _read_json(results / "mcp_surface.json")
        checklist = results / "mcp_checklist.md"
        self.checklist = checklist.read_text(encoding="utf-8") if checklist.is_file() else None
        self.written = (
            datetime.fromtimestamp((results / "cells.jsonl").stat().st_mtime)
            if (results / "cells.jsonl").is_file()
            else None
        )

    def score(self, cell: dict[str, Any]) -> dict[str, Any]:
        return self.scores.get(cell["cell_id"], {})

    def entry(self, cell: dict[str, Any]) -> dict[str, Any]:
        return self.entries.get(cell["entry"], {})

    def cell(self, cell_id: str) -> dict[str, Any] | None:
        return next((c for c in self.cells if c["cell_id"] == cell_id), None)

    def title(self, cell: dict[str, Any]) -> str | None:
        path = cell.get("transcript_path")
        if not path or not (self.results / path).is_file():
            return None
        return json.loads((self.results / path).read_text(encoding="utf-8")).get("title")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def load(results: Path) -> Data:
    return Data(results, load_manifest())


# --- hardware ---------------------------------------------------------------------------


def _first_match(path: str, pattern: str) -> str | None:
    try:
        found = re.search(pattern, Path(path).read_text(encoding="utf-8"), re.M)
    except OSError:
        return None
    return found.group(1).strip() if found else None


def _gpu() -> str | None:
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    fields = [p.strip() for p in done.stdout.strip().split("\n")[0].split(",")]
    if done.returncode != 0 or len(fields) != 3:  # e.g. "No devices were found"
        return None
    name, mem, driver = fields
    gib = round(int(mem.split()[0]) / 1024) if mem.split()[0].isdigit() else mem
    return f"{name} ({gib} GB, driver {driver})"


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


def hardware_line(short: bool = False) -> str:
    """The machine as it is when the report is rendered: the matrix ran here."""
    cpu = _first_match("/proc/cpuinfo", r"^model name\s*:\s*(.+)$") or platform.processor() or "?"
    threads = _first_match("/proc/cpuinfo", r"^siblings\s*:\s*(\d+)$")
    mem_kb = _first_match("/proc/meminfo", r"^MemTotal:\s*(\d+) kB$")
    parts = [
        _gpu() or "no CUDA GPU",
        f"{cpu} ({threads} threads)" if threads else cpu,
        f"{round(int(mem_kb) / 1024 / 1024)} GB RAM" if mem_kb else "RAM ?",
        f"faster-whisper {_version('faster-whisper')}",
    ]
    if not short:
        parts[3:3] = [
            f"{platform.system()} {platform.release()} {platform.machine()}",
            f"Python {platform.python_version()}",
        ]
        parts += [f"ctranslate2 {_version('ctranslate2')}", f"quotable {_version('quotable-mcp')}"]
    return " · ".join(parts)


# --- formatting -------------------------------------------------------------------------


def _code(cell_id: str) -> str:
    return "`" + cell_id.replace("|", "\\|") + "`"


def _s(value: Any, digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}f} s"


def _n(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _x(value: Any) -> str:
    return "—" if value is None else f"{value:.0f}×"


def _mb(value: Any) -> str:
    return "—" if value is None else f"{value:,.0f} MB".replace(",", " ")


def _pct(value: Any) -> str:
    return "—" if value is None else f"{value:+.1%}"


def _minutes(seconds: Any) -> str:
    if seconds is None:
        return "—"
    if seconds < 60:
        return f"{seconds:g} s"
    return f"{seconds / 60:.1f} min" if seconds < 3600 else f"{seconds / 3600:.1f} h"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _status(cell: dict[str, Any]) -> str:
    if cell["status"] == "ok":
        return "ok"
    return f"{cell['status']} ({cell['reason']})" if cell.get("reason") else cell["status"]


def _drift(score: dict[str, Any]) -> str:
    median = score.get("drift_median_s")
    if median is None:
        return "—"
    level = (score.get("drift_level") or "")[:1]
    bias = score.get("drift_bias_s")
    text = f"{median:.2f} s ({level})" if level else f"{median:.2f} s"
    return f"{text}, bias {bias:+.2f}" if bias is not None else text


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


# --- sections ---------------------------------------------------------------------------


def corpus_table(data: Data) -> str:
    rows = []
    for entry in data.entries.values():
        licence = entry.get("license") or {}
        name = licence.get("name", "—")
        url = licence.get("source_url")
        rows.append(
            [
                f"`{entry['id']}`",
                entry["kind"],
                _minutes(entry["expect"].get("duration_s")),
                f"{name} ([source]({url}))" if url and url.startswith("http") else name,
                f"`{entry['source']}`",
            ]
        )
    return _table(["Entry", "Kind", "Length", "Licence", "Source"], rows)


def _summary_cells(data: Data) -> list[dict[str, Any]]:
    return [
        c
        for c in data.cells
        if c["mode"] == "whisper"
        and c["status"] == "ok"
        and not c["diarize"]
        and not c["real_path"]
        and c["entry"] not in SUMMARY_EXCLUDE
        and (c.get("audio_s") or 0) >= SUMMARY_MIN_AUDIO_S
    ]


def summary(data: Data) -> str:
    counts = {"pass": 0, "fail": 0, "skipped": 0}
    for cell in data.cells:
        verdict = data.score(cell).get("pass")
        counts["skipped" if verdict is None else ("pass" if verdict else "fail")] += 1
    statuses: dict[str, int] = {}
    for cell in data.cells:
        statuses[cell["status"]] = statuses.get(cell["status"], 0) + 1
    lines = [
        f"{len(data.cells)} cells: {counts['pass']} pass, {counts['fail']} fail, "
        f"{counts['skipped']} skipped; statuses "
        + ", ".join(f"{k} {v}" for k, v in sorted(statuses.items()))
        + ".",
        "",
        f"Per model and device over the Whisper cells on cached audio of at least "
        f"{SUMMARY_MIN_AUDIO_S} s, without diarization and without "
        + ", ".join(f"`{e}`" for e in sorted(SUMMARY_EXCLUDE))
        + ". WER is normalised, against the oracle or the TTS script. VRAM is the peak over "
        "every cell of that model and device, diarize cells included.",
        "",
    ]
    rows = []
    pool = _summary_cells(data)
    for model in MODELS:
        for device in ("cpu", "cuda"):
            cells = [c for c in pool if c["model"] == model and c["device"] == device]
            wers = [data.score(c)["wer"] for c in cells if data.score(c).get("wer") is not None]
            vram = [
                c["vram_peak_mb"]
                for c in data.cells
                if c["model"] == model and c["device"] == device and c.get("vram_peak_mb")
            ]
            worst = max(
                ((data.score(c)["wer"], c["entry"]) for c in cells if data.score(c).get("wer")),
                default=None,
            )
            rows.append(
                [
                    f"`{model}`",
                    device,
                    str(len(cells)),
                    _n(_median(wers)),
                    f"{worst[0]:.3f} (`{worst[1]}`)" if worst else "—",
                    _x(_median([c["realtime"] for c in cells if c.get("realtime")])),
                    _s(_median([c["load_s"] for c in cells if c.get("load_s") is not None])),
                    _mb(max(vram)) if vram else "—",
                ]
            )
    lines.append(
        _table(
            [
                "Model",
                "Device",
                "Cells",
                "Median WER",
                "Worst WER",
                "Median realtime",
                "Load",
                "VRAM",
            ],
            rows,
        )
    )
    return "\n".join(lines)


def results_tables(data: Data) -> str:
    header = ["Cell", "Status", "WER", "WER raw", "Drift", "Realtime", "Load", "Cached", "Verdict"]
    out = []
    for kind, label in KINDS:
        cells = [c for c in data.cells if data.entry(c).get("kind") == kind]
        if not cells:
            continue
        rows = []
        for cell in cells:
            score = data.score(cell)
            verdict = PASS[score.get("pass")]
            if score.get("fails"):
                verdict += " " + ", ".join(f"`{f}`" for f in score["fails"])
            rows.append(
                [
                    _code(cell["cell_id"]),
                    _status(cell),
                    _n(score.get("wer")),
                    _n(score.get("wer_raw")),
                    _drift(score),
                    _x(cell.get("realtime")),
                    _s(cell.get("load_s")),
                    _s(cell.get("cached_s"), 2),
                    verdict,
                ]
            )
        out.append(f"#### {label}\n\n{_table(header, rows)}")
    return "\n\n".join(out)


def captions_table(data: Data) -> str:
    rows = []
    for cell in data.cells:
        if cell["mode"] != "captions":
            continue
        score, expect = data.score(cell), data.entry(cell).get("expect", {})
        kind = "transcript tag" if expect.get("transcript_tag") else expect.get("captions", "?")
        words = (
            f"{score['hyp_tokens']} / {score['ref_tokens']}"
            if score.get("ref_tokens") is not None
            else "—"
        )
        rows.append(
            [
                f"`{cell['entry']}`",
                kind,
                str(score.get("language") or "—"),
                words,
                _n(score.get("wer")),
                _drift(score),
                _s(cell.get("wall_s"), 2),
                PASS[score.get("pass")]
                + (" " + ", ".join(f"`{f}`" for f in score["fails"]) if score.get("fails") else ""),
            ]
        )
    if not rows:
        return "No captions cells."
    return _table(
        [
            "Entry",
            "Kind",
            "Language",
            "Words (captions / oracle)",
            "WER vs oracle",
            "Drift vs oracle",
            "Wall",
            "Verdict",
        ],
        rows,
    )


def diarize_table(data: Data) -> str:
    cells = [c for c in data.cells if c["diarize"]]
    if not cells:
        return "No diarize cells."
    if all(c["status"] == "skipped" for c in cells):
        return "Diarization: not run (" + ", ".join(sorted({c["reason"] for c in cells})) + ")."
    rows = []
    for cell in cells:
        score = data.score(cell)
        missed = (
            f"{score['turns_missed']} of {score['ref_turn_changes']}"
            if score.get("turns_missed") is not None
            else "—"
        )
        rows.append(
            [
                _code(cell["cell_id"]),
                _status(cell),
                str(score.get("hyp_speakers") if score.get("hyp_speakers") is not None else "—"),
                _n(score.get("speaker_accuracy")),
                missed,
                _n(score.get("wer")),
                _x(cell.get("realtime")),
                _mb(cell.get("vram_peak_mb")),
                PASS[score.get("pass")]
                + (" " + ", ".join(f"`{f}`" for f in score["fails"]) if score.get("fails") else ""),
            ]
        )
    return _table(
        [
            "Cell",
            "Status",
            "Speakers",
            "Speaker accuracy",
            "Turn changes missed",
            "WER",
            "Realtime",
            "VRAM",
            "Verdict",
        ],
        rows,
    )


def _scenario_numbers(name: str, record: dict[str, Any]) -> str:
    r = record
    if name == "straight":
        return (
            f"{_s(r.get('wall_s'))} for {_minutes(r.get('audio_s'))} ({_x(r.get('realtime'))}), "
            f"{r.get('segments')} segments, CUDA fallback {r.get('cuda_fallback')}"
        )
    if name == "memory":
        first = (r.get("samples") or [{}])[0]
        return (
            f"RSS {_pct(r.get('rss_growth'))}, VRAM {_pct(r.get('vram_growth'))} from "
            f"{_s(r.get('baseline_after_s'))}; first sample {_mb(first.get('rss_mb'))} RSS"
        )
    if name == "cancel_resume":
        return (
            f"cancelled with {r.get('partial_segments')} segments, lock free {r.get('lock_free')}, "
            f"resumed at {_s(r.get('first_resumed_start_s'))}, WER past the cancel "
            f"{_n(r.get('wer_past_cancel'))}, {r.get('words_lost_near_boundary')} lost / "
            f"{r.get('words_doubled_near_boundary')} doubled"
        )
    if name == "paging":
        return (
            f"{r.get('offset_pages')} offset pages exact {r.get('offset_text_exact')}; "
            f"{r.get('windows')} windows / {r.get('window_pages')} pages, "
            f"{len(r.get('window_mismatches_at_s') or [])} mismatches, "
            f"{r.get('segments_in_no_window')} in no window, {r.get('segments_in_two_windows')} in "
            f"two of {r.get('segments_straddling_a_boundary')} straddling"
        )
    if name == "feed_transcript":
        return (
            f"method {r.get('method')}, {r.get('enclosure_downloads')} enclosure downloads, "
            f"{r.get('segments')} segments, {r.get('words')} words"
        )
    return ", ".join(f"{k} {v}" for k, v in r.items() if k not in ("pass", "samples"))


def scenarios_table(data: Data) -> str:
    if not data.scenarios:
        return "Scenarios: not run."
    rows = []
    for entry, found in data.scenarios.items():
        for name, record in found.items():
            if name == "error" or not isinstance(record, dict):
                rows.append([f"`{entry}`", name, "FAIL", str(record)])
                continue
            rows.append(
                [
                    f"`{entry}`",
                    f"`{name}`",
                    PASS[record.get("pass")],
                    _scenario_numbers(name, record),
                ]
            )
    return _table(["Entry", "Scenario", "Verdict", "Measured"], rows)


_CHECK_ITEM = re.compile(r"^- \[(?P<box>[ x])\] (?P<fail>FAIL: )?(?P<title>[^—]+)—(?P<rest>.*)$")


def mcp_section(data: Data) -> str:
    lines = []
    if data.surface:
        info, rt = data.surface.get("server_info", {}), data.surface.get("round_trip", {})
        tools = ", ".join(f"`{t['name']}`" for t in data.surface.get("tools", []))
        templates = ", ".join(
            f"`{t['uriTemplate']}`" for t in data.surface.get("resource_templates", [])
        )
        lines += [
            f"Surface of `{info.get('name')}` {info.get('version')}: tools {tools}; resource "
            f"templates {templates}; instructions "
            f"{'present' if data.surface.get('instructions') else 'missing'}.",
            "",
            f"Round trip on the built wheel: **{'ok' if rt.get('ok') else 'FAILED'}** "
            f"(`{' '.join(rt.get('command', []))}`), checks "
            + ", ".join(f"{k} {v}" for k, v in (rt.get("checks") or {}).items())
            + ".",
        ]
    else:
        lines.append("MCP probe: not run.")
    if data.checklist:
        rows, observed = [], ""
        for line in data.checklist.splitlines():
            m = _CHECK_ITEM.match(line)
            if m:
                if rows:
                    rows[-1][2] = observed.strip()
                box, fail = m.group("box") == "x", bool(m.group("fail"))
                verdict = "FAIL" if fail else ("pass" if box else "open")
                rows.append([m.group("title").strip(), verdict, ""])
                observed = ""
            elif rows and line.strip():
                observed += " " + line.strip().removeprefix("Observed:").strip()
        if rows:
            rows[-1][2] = observed.strip()
        lines += [
            "",
            "In-client checklist (Claude Code):",
            "",
            _table(["Item", "Verdict", "Observed"], rows),
        ]
    else:
        lines += ["", "In-client checklist: not sat."]
    return "\n".join(lines)


def _results_part(rendered: str) -> str:
    """The rendered tables without the first line, which describes the machine and the
    results file's mtime, so --check does not go stale on a driver update or another box."""
    return rendered.strip().partition("\n\n")[2]


def render(data: Data) -> str:
    written = data.written.strftime("%Y-%m-%d %H:%M") if data.written else "—"
    sections = [
        f"Hardware: {hardware_line()}. Results written {written}; rendered by "
        "`python -m evals.report`.",
        "### Corpus\n\n" + corpus_table(data),
        "### Summary\n\n" + summary(data),
        "### Matrix\n\n" + results_tables(data),
        "### Captions against the oracle\n\n" + captions_table(data),
        "### Diarization\n\n" + diarize_table(data),
        "### Scenarios\n\n" + scenarios_table(data),
        "### MCP\n\n" + mcp_section(data),
    ]
    return "\n\n".join(sections) + "\n"


# --- README -----------------------------------------------------------------------------


def _label(data: Data, cell: dict[str, Any], entry: dict[str, Any]) -> str:
    kind = {"feed": "podcast feed", "youtube": "YouTube", "file": "local file", "tts": "local file"}
    # The cached-audio cells are titled after the downloaded file; the live cell has the real one.
    live = data.cell(f"{entry['id']}|whisper|turbo|cuda|false|true")
    title = (data.title(live) if live else None) or data.title(cell) or entry["id"]
    title = title.replace("|", "\\|")
    return f"*{title}* ({kind.get(entry['kind'], entry['kind'])}, {_minutes(cell.get('audio_s'))})"


def _captioned(entry: dict[str, Any]) -> bool:
    expect = entry["expect"]
    return expect.get("captions") not in (None, "none") or bool(expect.get("transcript_tag"))


def readme_section(data: Data) -> str:
    rows, notes = [], []
    wanted = [e for e in data.entries.values() if e.get("readme_row")]
    for entry in sorted(wanted, key=_captioned):  # Whisper rows first, then the captions row
        if _captioned(entry):
            cell = data.cell(f"{entry['id']}|captions|-|-|false|false")
            if cell and cell["status"] == "ok":
                rows.append(
                    [
                        _label(data, cell, entry),
                        "captions",
                        "-",
                        _minutes(cell["audio_s"]),
                        "-",
                        _s(cell["wall_s"]),
                        _x(cell["realtime"]),
                        _s(cell["cached_s"], 2),
                    ]
                )
            continue
        first = True
        for model, device in README_WHISPER:
            cell = data.cell(f"{entry['id']}|whisper|{model}|{device}|false|false")
            if not cell or cell["status"] != "ok":
                continue
            rows.append(
                [
                    _label(data, cell, entry) if first else "same episode",
                    model,
                    device,
                    _minutes(cell["audio_s"]),
                    _s(cell["load_s"]),
                    _s(cell["wall_s"]),
                    _x(cell["realtime"]),
                    _s(cell["cached_s"], 2),
                ]
            )
            first = False
        live = data.cell(f"{entry['id']}|whisper|turbo|cuda|false|true")
        if live and live["status"] == "ok":
            notes.append(
                f"Fetched live from the feed, the `turbo` row took {_s(live['wall_s'])} including "
                "the download."
            )
    when = data.written.strftime("%B %Y") if data.written else "—"
    text = [
        README_SECTION,
        "",
        f"Measured with `python -m evals.run` ({when}) on {hardware_line(short=True)}.",
        "Audio is the decoded length; load is the model load; wall is one `transcribe` call",
        "after that, on the already downloaded audio; realtime is audio divided by wall; cached",
        "is a second call for the same variant.",
        "",
        _table(["Source", "Model", "Device", "Audio", "Load", "Wall", "Realtime", "Cached"], rows),
        "",
    ]
    if notes:
        text += [" ".join(notes), ""]
    text.append(
        "The full matrix (every input kind, four models, both devices, the long-run scenarios) is "
        "in [`evals/README.md`](evals/README.md)."
    )
    return "\n".join(text) + "\n"


def _split_readme(text: str) -> tuple[str, str, str]:
    """``(before, section, after)`` around the Benchmark section, up to the next ``## ``."""
    start = text.find(README_SECTION + "\n")
    if start < 0:
        raise ValueError(f"README has no `{README_SECTION}` section")
    end = text.find("\n## ", start + len(README_SECTION))
    end = len(text) if end < 0 else end + 1
    return text[:start], text[start:end], text[end:]


def rewrite_readme(data: Data, readme: Path = README) -> None:
    before, _, after = _split_readme(readme.read_text(encoding="utf-8"))
    readme.write_text(before + readme_section(data) + "\n" + after, encoding="utf-8")


# --- the doc block ----------------------------------------------------------------------


def _split_block(text: str) -> tuple[str, str, str] | None:
    start, end = text.find(TABLES_START), text.find(TABLES_END)
    if start < 0 or end < 0 or end < start:
        return None
    start += len(TABLES_START)
    return text[:start], text[start:end], text[end:]


def update_doc(doc: Path, tables: str) -> bool:
    """Replace the tables block in ``doc``; False when the file or its markers are absent."""
    if not doc.is_file():
        return False
    parts = _split_block(doc.read_text(encoding="utf-8"))
    if parts is None:
        return False
    before, _, after = parts
    doc.write_text(before + "\n" + tables + after, encoding="utf-8")
    return True


# --- check ------------------------------------------------------------------------------


OLD_SCRIPT = "bench" + ".py"  # spelled in two parts so this file does not match its own check


def _bench_mentions() -> list[str]:
    out = subprocess.run(
        ["git", "grep", "-l", OLD_SCRIPT, "--", ".", ":!docs/development", ":!docs/.archive"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    return out.stdout.split()


def check(data: Data, doc: Path = DOC, readme: Path = README) -> list[str]:
    problems = []
    if not data.cells:
        problems.append(f"no rows in {data.results / 'cells.jsonl'}")
    for name, present in (
        ("scenarios.json", data.scenarios),
        ("mcp_surface.json", data.surface),
        ("mcp_checklist.md", data.checklist),
    ):
        if not present:
            problems.append(f"{data.results / name} missing: the report is incomplete")
    if not doc.is_file():
        problems.append(f"{doc} missing")
    else:
        text = doc.read_text(encoding="utf-8")
        parts = _split_block(text)
        if parts is None:
            problems.append(f"{doc} has no {TABLES_START} … {TABLES_END} block")
        elif _results_part(parts[1]) != _results_part(render(data)):
            problems.append(f"{doc}: tables are stale (re-run python -m evals.report)")
        plain = text.replace("\\|", "|")
        problems += [
            f"{c['cell_id']}: not in {doc.name}" for c in data.cells if c["cell_id"] not in plain
        ]
    if not readme.is_file():
        problems.append(f"{readme} missing")
    else:
        text = readme.read_text(encoding="utf-8")
        try:
            section = _split_readme(text)[1]
        except ValueError as exc:
            problems.append(str(exc))
        else:
            if "pending" in section:
                problems.append(f"{readme.name}: a benchmark row is still pending")
        if "TODO(bench)" in text:
            problems.append(f"{readme.name}: TODO(bench) is still there")
    problems += [f"{path} mentions {OLD_SCRIPT}" for path in _bench_mentions()]
    return problems


# --- main -------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals.report", description=__doc__)
    parser.add_argument("--results", default=str(RESULTS), help="directory with cells.jsonl")
    parser.add_argument("--doc", default=str(DOC), help="the report whose tables block to update")
    parser.add_argument(
        "--readme", action="store_true", help="rewrite the README benchmark section"
    )
    parser.add_argument("--check", action="store_true", help="replay the files; render nothing")
    args = parser.parse_args(argv)
    data = load(Path(args.results).resolve())
    doc = Path(args.doc).resolve()

    if args.check:
        problems = check(data, doc)
        for p in problems:
            print(p)
        print(f"report: {len(problems)} problems")
        return 1 if problems else 0
    if args.readme:
        rewrite_readme(data)
        print(f"{README.relative_to(ROOT)}: Benchmark section rewritten")
        return 0
    tables = render(data)
    (data.results / "report.md").write_text(tables, encoding="utf-8")
    print(f"{len(data.cells)} cells -> {(data.results / 'report.md').relative_to(ROOT)}")
    if update_doc(doc, tables):
        print(f"{doc.relative_to(ROOT)}: tables block updated")
    else:
        print(f"{doc.relative_to(ROOT)}: no tables block to update ({TABLES_START})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
