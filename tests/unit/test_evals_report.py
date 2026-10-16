"""The eval report on a synthetic results directory: tables, the README section, --check."""

import json

import pytest

from evals import report as rp

ENTRIES = [
    {
        "id": "clip",
        "kind": "file",
        "source": "clip.wav",
        "license": {"name": "Own recording", "source_url": "notes.md"},
        "expect": {"captions": "none", "transcript_tag": False, "duration_s": 1800.0},
        "readme_row": True,
    },
    {
        "id": "talk",
        "kind": "youtube",
        "source": "https://www.youtube.com/watch?v=x",
        "license": {"name": "CC BY", "source_url": "https://example.org/terms"},
        "expect": {"captions": "manual", "transcript_tag": False, "duration_s": 600.0},
        "readme_row": True,
    },
]


def _cell(cell_id, **over):
    entry, mode, model, device, diarize, real = cell_id.split("|")
    row = {
        "cell_id": cell_id,
        "entry": entry,
        "mode": mode,
        "model": None if model == "-" else model,
        "device": None if device == "-" else device,
        "diarize": diarize == "true",
        "real_path": real == "true",
        "status": "ok",
        "reason": None,
        "wall_s": 60.0,
        "load_s": None if mode == "captions" else 1.0,
        "audio_s": 1800.0,
        "realtime": 30.0,
        "cached_s": 0.02,
        "vram_peak_mb": 2000 if device == "cuda" else None,
        "max_rss_mb": 1000.0,
        "transcript_path": None,
    }
    row.update(over)
    return row


def _score(cell_id, **over):
    row = {
        "cell_id": cell_id,
        "pass": True,
        "fails": [],
        "wer": 0.05,
        "wer_raw": 0.08,
        "drift_level": "segment",
        "drift_median_s": 0.3,
        "drift_bias_s": -0.1,
        "language": "en",
        "ref_tokens": 100,
        "hyp_tokens": 101,
    }
    row.update(over)
    return row


CELLS = [
    _cell("clip|whisper|base|cpu|false|false"),
    _cell("clip|whisper|turbo|cuda|false|false", wall_s=30.0, realtime=60.0),
    _cell("clip|whisper|turbo|cuda|false|true", wall_s=35.0),
    _cell("talk|captions|-|-|false|false", audio_s=600.0, wall_s=2.0, realtime=300.0),
    _cell("talk|whisper|turbo|cuda|true|false", audio_s=600.0),
]
SCORES = [
    _score("clip|whisper|base|cpu|false|false"),
    _score("clip|whisper|turbo|cuda|false|false", pass_=None),
    _score("clip|whisper|turbo|cuda|false|true"),
    _score("talk|captions|-|-|false|false", pass_=False, fails=["language"], language="sq"),
    _score("talk|whisper|turbo|cuda|true|false", speaker_accuracy=0.95, hyp_speakers=2),
]
for s in SCORES:  # `pass` is a keyword, so the helper takes `pass_`
    if "pass_" in s:
        s["pass"] = s.pop("pass_")


@pytest.fixture
def results(tmp_path, monkeypatch):
    out = tmp_path / "results"
    out.mkdir()
    (out / "cells.jsonl").write_text("".join(json.dumps(c) + "\n" for c in CELLS))
    (out / "scores.jsonl").write_text("".join(json.dumps(s) + "\n" for s in SCORES))
    (out / "scenarios.json").write_text(
        json.dumps(
            {
                "clip": {
                    "straight": {"pass": True, "wall_s": 30.0, "audio_s": 1800.0, "realtime": 60.0},
                    "memory": {"pass": False, "rss_growth": 0.2, "vram_growth": 0.0, "samples": []},
                }
            }
        )
    )
    (out / "mcp_surface.json").write_text(
        json.dumps(
            {
                "server_info": {"name": "quotable", "version": "0.1.0"},
                "instructions": "x",
                "tools": [{"name": "transcribe"}],
                "resource_templates": [{"uriTemplate": "transcript://{id}"}],
                "round_trip": {"ok": True, "command": ["quotable", "serve"], "checks": {"a": True}},
            }
        )
    )
    (out / "mcp_checklist.md").write_text(
        "# Checklist\n\n- [x] Connected — the server shows up.\n  Observed: yes.\n"
        "- [x] FAIL: Mention — `@` attaches.\n  Observed: no.\n- [ ] Open — later.\n  Observed:\n"
    )
    monkeypatch.setattr(rp, "hardware_line", lambda short=False: "test box")
    monkeypatch.setattr(rp, "_bench_mentions", lambda: [])
    return out


@pytest.fixture
def data(results):
    return rp.Data(results, ENTRIES)


# --- tables -----------------------------------------------------------------------------


def test_render_names_every_cell_and_escapes_the_pipes(data):
    text = rp.render(data)
    for cell in CELLS:
        assert "`" + cell["cell_id"].replace("|", "\\|") + "`" in text
    assert "#### Local files" in text and "#### YouTube" in text
    assert "([source](https://example.org/terms))" in text  # licence link only for URLs
    assert "Own recording" in text and "notes.md" not in text


def test_summary_counts_pass_fail_and_skipped(data):
    text = rp.summary(data)
    assert "5 cells: 3 pass, 1 fail, 1 skipped" in text


def test_captions_and_diarize_tables_carry_their_verdicts(data):
    captions = rp.captions_table(data)
    assert "| `talk` | manual | sq | 101 / 100 |" in captions
    assert "FAIL `language`" in captions
    diarize = rp.diarize_table(data)
    assert "| 2 | 0.950 |" in diarize


def test_diarize_table_says_not_run_when_every_cell_was_skipped(results):
    cells = [_cell("talk|whisper|turbo|cuda|true|false", status="skipped", reason="no token")]
    (results / "cells.jsonl").write_text(json.dumps(cells[0]) + "\n")
    assert rp.diarize_table(rp.Data(results, ENTRIES)) == "Diarization: not run (no token)."


def test_scenarios_and_checklist_render_pass_and_fail(data):
    scenarios = rp.scenarios_table(data)
    assert "| `clip` | `straight` | pass |" in scenarios
    assert "| `clip` | `memory` | FAIL | RSS +20.0%" in scenarios
    mcp = rp.mcp_section(data)
    assert "| Connected | pass | yes. |" in mcp
    assert "| Mention | FAIL | no. |" in mcp
    assert "| Open | open |  |" in mcp
    assert "Round trip on the built wheel: **ok**" in mcp


def test_memory_row_without_a_vram_reading_renders_a_dash(results):
    # growth() is None with no nvidia-smi reading for the process (CPU run, no GPU)
    memory = {"pass": True, "rss_growth": 0.01, "vram_growth": None, "samples": []}
    (results / "scenarios.json").write_text(json.dumps({"clip": {"memory": memory}}))
    assert "RSS +1.0%, VRAM — from" in rp.scenarios_table(rp.Data(results, ENTRIES))


def test_gpu_is_none_when_nvidia_smi_reaches_no_gpu(monkeypatch):
    done = rp.subprocess.CompletedProcess([], 6, stdout="No devices were found\n", stderr="")
    monkeypatch.setattr(rp.subprocess, "run", lambda *a, **k: done)
    assert rp._gpu() is None


# --- README -----------------------------------------------------------------------------


def test_readme_section_has_the_whisper_rows_then_the_captions_row(data):
    section = rp.readme_section(data)
    lines = [line for line in section.splitlines() if line.startswith(("| *", "| same"))]
    assert lines[0].startswith("| *clip* (local file, 30.0 min) | base | cpu |")
    assert lines[1].startswith("| same episode | turbo | cuda |")
    assert lines[2].startswith("| *talk* (YouTube, 10.0 min) | captions | - |")
    assert "took 35.0 s including the download" in section
    assert "pending" not in section


def test_rewrite_readme_replaces_only_the_benchmark_section(data, tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# x\n\n## Benchmark\n\nold | pending |\n\n## Next\n\nkeep\n")
    rp.rewrite_readme(data, readme)
    text = readme.read_text()
    assert "pending" not in text
    assert text.startswith("# x\n\n## Benchmark\n\nMeasured with")
    assert text.endswith("\n## Next\n\nkeep\n")


# --- the doc block and --check ----------------------------------------------------------


def _doc(tmp_path, body="x"):
    doc = tmp_path / "quotable-eval.md"
    doc.write_text(f"# Report\n\nprose\n\n{rp.TABLES_START}\n{body}\n{rp.TABLES_END}\n")
    return doc


def test_update_doc_replaces_the_block_and_check_passes(data, tmp_path):
    doc = _doc(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text("## Benchmark\n\nfine\n")
    assert rp.update_doc(doc, rp.render(data))
    assert doc.read_text().startswith("# Report\n\nprose\n\n")
    assert rp.check(data, doc, readme) == []


def test_check_ignores_the_hardware_line(data, tmp_path, monkeypatch):
    doc = _doc(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text("## Benchmark\n\nfine\n")
    rp.update_doc(doc, rp.render(data))
    monkeypatch.setattr(rp, "hardware_line", lambda short=False: "another box, new driver")
    assert rp.check(data, doc, readme) == []


def test_check_reports_stale_tables_a_pending_row_and_missing_cells(data, tmp_path):
    doc = _doc(tmp_path, "stale")
    readme = tmp_path / "README.md"
    readme.write_text("## Benchmark\n\n| pending |\n\n<!-- TODO(bench) -->\n")
    problems = rp.check(data, doc, readme)
    assert any("stale" in p for p in problems)
    assert sum("not in quotable-eval.md" in p for p in problems) == len(CELLS)
    assert any("still pending" in p for p in problems)
    assert any("TODO(bench)" in p for p in problems)


def test_check_wants_the_markers_and_the_side_files(data, tmp_path, results):
    doc = tmp_path / "quotable-eval.md"
    doc.write_text("no markers\n")
    (results / "scenarios.json").unlink()
    problems = rp.check(rp.Data(results, ENTRIES), doc, tmp_path / "missing-README.md")
    assert any("no <!-- MERGE_START(TABLES) -->" in p for p in problems)
    assert any("scenarios.json missing" in p for p in problems)
    assert any("missing-README.md missing" in p for p in problems)


def test_update_doc_is_false_without_the_file_or_the_markers(data, tmp_path):
    assert not rp.update_doc(tmp_path / "none.md", "t")
    plain = tmp_path / "plain.md"
    plain.write_text("nothing\n")
    assert not rp.update_doc(plain, "t")
    assert plain.read_text() == "nothing\n"
